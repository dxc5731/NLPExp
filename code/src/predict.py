# -*- coding: utf-8 -*-
"""用训练好的模型跑一个划分（默认 dev），产出交接用的预测文件和 run_info.json。

用法（在 code/ 目录下）：
    python -X utf8 -m src.predict --run-id lr5e-5_ep3_20261022-151229
    python -X utf8 -m src.predict --run-id xxx --split dev --batch-size 64

产物（默认写在 runs/<run_id>/predict/ 下）：
    predictions_dev.json   两层字典：对话ID -> 句子ID -> BIO 字符串
    run_info.json          这次推理的可复现信息
    inference.log          推理过程的文字记录

几条硬规矩（来自《预测结果交接规范》）：
    - 原始对话 ID 和句子 ID 原样保留，不重新编号
    - 每个原始字符一个标签，数量必须等于 len(sentence)
    - 只用规定的 11 种标签，单个 ASCII 空格分隔
    - 不输出 [CLS]/[SEP]/补位
    - 非法 BIO 会被修复，但修复处数记进日志，不隐瞒
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from .align import to_legal_bio
from .config import CODE, DATA_DIR, MAX_LEN, MODEL_DIR, ROOT, RUNS
from .data_io import load_split, sha256_file, split_path
from .dataset import build_loader, collect_examples, load_tokenizer
from .labels import LABELS
from .metrics import decode_batch
from .model import build_model

ARCHIVE_SHA256 = "84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67"


def source_snapshot_sha256() -> str:
    """没有 git 时，用源码文件的哈希代表代码版本。"""
    h = hashlib.sha256()
    for p in sorted((CODE / "src").glob("*.py")):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def git_info() -> dict:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                        stderr=subprocess.DEVNULL).decode().strip()
        return {"git_revision": rev, "git_dirty": bool(dirty)}
    except Exception:
        return {
            "git_revision": None,
            "git_dirty": None,
            "source_snapshot": "code/src/*.py",
            "source_sha256": source_snapshot_sha256(),
            "note": "运行环境没有 git 仓库，用源码文件哈希代表代码版本",
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    run_dir = RUNS / args.run_id
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        sys.exit(f"找不到模型：{ckpt_path}")

    out_dir = Path(args.output_dir) if args.output_dir else run_dir / f"predict_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "inference.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    t_start = time.time()
    log(f"run_id={args.run_id}")
    log(f"checkpoint={ckpt_path}")
    log(f"split={args.split}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_cfg = ckpt.get("config", {})
    log(f"训练配置: {json.dumps(train_cfg, ensure_ascii=False)}")
    log(f"该 checkpoint 来自第 {ckpt.get('epoch')} 轮，验证集实体 F1 {ckpt.get('metrics', {}).get('entity_f1')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer()
    data = load_split(args.split)
    examples = collect_examples(data, None, require_labels=False)
    log(f"句子数 {len(examples)}，对话数 {len(data)}")

    loader = build_loader(examples, tokenizer, batch_size=args.batch_size,
                          shuffle=False, max_len=MAX_LEN, with_labels=False)

    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    predictions = {}
    n_repair = 0
    n_done = 0
    t_infer = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            pred_ids = logits.argmax(dim=-1).cpu()
            decoded, repaired = decode_batch(pred_ids, batch["offsets"], batch["text"])
            n_repair += repaired
            for i, tags in enumerate(decoded):
                did = batch["dialogue_id"][i]
                sid = batch["sentence_id"][i]
                predictions.setdefault(did, {})[sid] = " ".join(tags)
                n_done += 1
            if n_done % 5000 < args.batch_size:
                log(f"  已处理 {n_done}/{len(examples)} 句")
    infer_seconds = time.time() - t_infer

    # ---- 自检：ID 全覆盖、长度一致、标签合法 ----
    assert set(predictions) == set(data), "预测的对话 ID 集合与原始数据不一致"
    n_sent = 0
    for did, d in data.items():
        assert set(predictions[did]) == {s["sentence_id"] for s in d["dialogue"]}, f"对话 {did} 句子 ID 不一致"
        for s in d["dialogue"]:
            tags = predictions[did][s["sentence_id"]].split(" ")
            assert len(tags) == len(s["sentence"]), f"{did}/{s['sentence_id']} 标签数与字符数不一致"
            assert all(t in LABELS for t in tags), f"{did}/{s['sentence_id']} 出现非法标签"
            n_sent += 1
    log(f"自检通过：{len(predictions)} 段对话 / {n_sent} 句，标签数与字符数全部一致")
    log(f"非法 BIO 修复 {n_repair} 处")

    pred_path = out_dir / f"predictions_{args.split}.json"
    with open(pred_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(predictions, f, ensure_ascii=False)
    log(f"预测写入 {pred_path}  ({pred_path.stat().st_size / 1e6:.1f} MB)")

    elapsed = time.time() - t_start
    run_info = {
        "schema_version": "1.0",
        "artifact_kind": "model_prediction",
        "run_id": args.run_id,
        "method": {
            "name": "BERT + Softmax（逐位置分类，无 CRF）",
            "kind": "bert",
            "method_version": f"checkpoint: runs/{args.run_id}/checkpoints/best.pt",
        },
        "data": {
            "version": "IMCS-V2-NER（CBLUE）",
            "split": args.split,
            "source_file": str(split_path(args.split)),
            "source_sha256": sha256_file(split_path(args.split)),
            "archive_sha256": ARCHIVE_SHA256,
            "archive_sha256_note": "取自《预测结果交接规范》给定的原始 ZIP 固定哈希；"
                                   "本机使用解压后的 JSON，未直接核对 ZIP 字节",
        },
        "code": git_info(),
        "config": {
            "training_config": train_cfg,
            "max_len": MAX_LEN,
            "label_set": LABELS,
        },
        "preprocessing": {
            "tokenizer": "bert-base-chinese（WordPiece，fast）",
            "truncation": "无。最长句 201 词片，上限 256，超长会直接报错而不是截断",
            "special_tokens": "[CLS]/[SEP]/补位不参与预测，也不出现在输出里",
        },
        "alignment": {
            "rule": "词片预测按 offset 摊回原始字符；一个词片覆盖多个字时首字用原标签、"
                    "其余 B-X 降级为 I-X",
            "illegal_bio_repairs": n_repair,
            "note": "修复只作用于预测结果；对齐校验路径不做任何修复",
        },
        "prediction": {
            "file": pred_path.name,
            "sha256": sha256_file(pred_path),
            "dialogues": len(predictions),
            "sentences": n_sent,
        },
        "inference_command": (
            f"python -X utf8 -m src.predict --run-id {args.run_id} --split {args.split} "
            f"--batch-size {args.batch_size}"
        ),
        "inference_working_directory": "code/",
        "device": str(device),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "inference_elapsed_seconds": round(elapsed, 2),
        "inference_pure_seconds": round(infer_seconds, 2),
        "timing_scope": "含模型加载、分词、推理和写盘；不含训练",
        "random_seeds": {"seed": train_cfg.get("seed"), "note": "推理无随机过程，argmax 确定性"},
        "training": {
            "train_split": "code/artifacts/split_v1.json 的 train 列表（2225 段对话）",
            "command": f"python -X utf8 -m src.train --run-id {args.run_id}",
            "config_path": f"runs/{args.run_id}/config.json",
            "log_path": f"runs/{args.run_id}/train.log",
            "checkpoint": f"runs/{args.run_id}/checkpoints/best.pt",
            "checkpoint_sha256": sha256_file(ckpt_path),
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    info_path = out_dir / "run_info.json"
    with open(info_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    log(f"run_info 写入 {info_path}")
    log(f"总耗时 {elapsed:.1f} 秒（纯推理 {infer_seconds:.1f} 秒）")


if __name__ == "__main__":
    main()
