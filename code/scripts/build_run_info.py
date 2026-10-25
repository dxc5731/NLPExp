# -*- coding: utf-8 -*-
"""按评估模块的真实 schema 生成 run_info.json。

schema 以 evaluation 分支里的 src/check_run_info.py 为准（不是凭交接规范散文猜的）。

用法（在 code/ 目录下）：
    python -X utf8 -m scripts.build_run_info --run-id lr5e-5_ep3_20261022-151229

产物写到 runs/<run_id>/deliver/，目录结构：
    predictions_dev.json
    run_info.json
    config/config.json
    logs/train.log, logs/inference.log
    snapshot/source_snapshot.zip
"""

import argparse
import hashlib
import json
import platform
import shutil
import time
import zipfile
from pathlib import Path

import torch

from src.config import CODE, DATA_DIR, RUNS
from src.data_io import sha256_file

METHOD_NAME = "bert_softmax"
ARCHIVE_SHA256 = "84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67"


def make_snapshot(path: Path) -> None:
    """把源码打成 zip，作为「代码快照」交给评估方核对。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((CODE / "src").glob("*.py")):
            z.write(p, f"src/{p.name}")
        for p in sorted((CODE / "scripts").glob("*.py")):
            z.write(p, f"scripts/{p.name}")
        z.write(CODE / "requirements.txt", "requirements.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--predict-dir", default=None,
                    help="已有预测所在目录；默认 runs/<run_id>/predict_dev")
    ap.add_argument("--env-from", default=None,
                    help="从哪份旧 run_info 读取真实设备/耗时；默认同 predict-dir")
    args = ap.parse_args()

    run_dir = RUNS / args.run_id
    predict_dir = Path(args.predict_dir) if args.predict_dir else run_dir / f"predict_{args.split}"
    env_from = Path(args.env_from) if args.env_from else predict_dir / "run_info.json"

    old = json.loads(env_from.read_text(encoding="utf-8")) if env_from.exists() else {}
    # 权重在服务器上，本地只有配置和日志；配置从 run 目录的 config.json 读
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
    ckpt_sha = (old.get("training", {}) or {}).get("checkpoint_sha256")

    out = run_dir / ("deliver" if args.split == "dev" else f"deliver_{args.split}")
    if out.exists():
        shutil.rmtree(out)
    (out / "config").mkdir(parents=True)
    (out / "logs").mkdir(parents=True)
    (out / "snapshot").mkdir(parents=True)

    pred_src = predict_dir / f"predictions_{args.split}.json"
    shutil.copy2(pred_src, out / f"predictions_{args.split}.json")
    shutil.copy2(run_dir / "config.json", out / "config" / "config.json")
    shutil.copy2(run_dir / "train.log", out / "logs" / "train.log")
    shutil.copy2(predict_dir / "inference.log", out / "logs" / "inference.log")
    make_snapshot(out / "snapshot" / "source_snapshot.zip")

    device = (old.get("environment", {}).get("gpu")
              or (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
    old_env = old.get("environment", {})
    info = {
        "schema_version": "1.0",
        "artifact_kind": "model_prediction",
        "run_id": args.run_id,
        "method": {"name": METHOD_NAME, "kind": "bert"},
        "method_version": (
            f"bert-base-chinese + 11 类线性分类头；"
            f"checkpoint runs/{args.run_id}/checkpoints/best.pt"
            + (f" sha256={ckpt_sha}" if ckpt_sha else "")
        ),
        "data": {
            "version": "IMCS-V2-NER-platform-84e34d24",
            "split": args.split,
            "source_file": f"IMCS-V2-NER/IMCS-V2_{args.split}.json",
            "source_sha256": sha256_file(DATA_DIR / f"IMCS-V2_{args.split}.json"),
            "archive_sha256": ARCHIVE_SHA256,
        },
        "code": {
            "revision": None,
            "dirty": None,
            "snapshot": {
                "path": "snapshot/source_snapshot.zip",
                "sha256": sha256_file(out / "snapshot" / "source_snapshot.zip"),
            },
        },
        "config": {
            "version": f"bert-softmax-lr{cfg.get('lr')}-ep{cfg.get('epochs')}-seed{cfg.get('seed')}",
            "path": "config/config.json",
            "sha256": sha256_file(out / "config" / "config.json"),
            "inline": cfg,
        },
        "logs": [
            {"path": "logs/inference.log", "sha256": sha256_file(out / "logs" / "inference.log")},
            {"path": "logs/train.log", "sha256": sha256_file(out / "logs" / "train.log")},
        ],
        "preprocessing": (
            "只做 WordPiece 分词，不改原文：不删空格、不归一化、不拼 speaker。"
            "最长句 201 词片，上限 256，超长直接报错而不是截断。"
        ),
        "alignment": {
            "tokenizer": "bert-base-chinese（WordPiece fast tokenizer），vocab 21128",
            "character_mapping": (
                "用 tokenizer 返回的 offset 把词片映射回原始字符；"
                "一个词片覆盖多个字时首字用原标签、其余 B-X 降级为 I-X。全量 98529 句校验，"
                "非歧义句子逐字还原一致，明细见 code/artifacts/alignment_report_*.json"
            ),
            "special_tokens": "[CLS]/[SEP]/补位不参与预测，也不出现在输出里；导出前只对预测结果修非法 BIO，并计数",
            "long_text": "本数据集最长 199 字符，无需滑窗；encode 时关闭 truncation，超长会抛错",
        },
        "prediction": {
            "file": f"predictions_{args.split}.json",
            "sha256": sha256_file(out / f"predictions_{args.split}.json"),
        },
        "inference_command": (
            f"python -X utf8 -m src.predict --run-id {args.run_id} --split {args.split} --batch-size 64"
        ),
        "inference_working_directory": "code/",
        "device": device,
        "environment": {
            "python": old_env.get("python") or platform.python_version(),
            "packages": {
                "torch": old_env.get("torch") or torch.__version__,
                "transformers": "5.20.1",
                "tokenizers": "0.23.3",
                "seqeval": "1.2.2",
                "numpy": "2.4.6",
            },
        },
        "inference_elapsed_seconds": old.get("inference_elapsed_seconds", 0.0),
        "timing_scope": "含模型加载、分词、推理和写盘；不含训练",
        "random_seeds": {"python": int(cfg.get("seed", 42)),
                         "numpy": int(cfg.get("seed", 42)),
                         "torch": int(cfg.get("seed", 42))},
        "training": {
            "model_version": f"runs/{args.run_id}/checkpoints/best.pt",
            "command": (
                f"python -X utf8 -m src.train --lr {cfg.get('lr')} --epochs {cfg.get('epochs')} "
                f"--seed {cfg.get('seed')} --run-id {args.run_id}"
            ),
            "data": [{
                "split": "train",
                "path": "../../../data/IMCS-V2_train.json",
                "sha256": sha256_file(DATA_DIR / "IMCS-V2_train.json"),
            }],
            "config": {"path": "config/config.json",
                       "sha256": sha256_file(out / "config" / "config.json")},
            "logs": [{"path": "logs/train.log",
                      "sha256": sha256_file(out / "logs" / "train.log")}],
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    (out / "run_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"写入 {out / 'run_info.json'}")
    print(f"交付目录 {out}")


if __name__ == "__main__":
    main()
