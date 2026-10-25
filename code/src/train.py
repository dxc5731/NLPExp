# -*- coding: utf-8 -*-
"""BERT + Softmax 训练脚本。

用法（在 code/ 目录下）：
    python -m src.train                                  # 默认配置跑 3 轮
    python -m src.train --lr 2e-5 --epochs 3             # 换学习率
    python -m src.train --limit-dialogues 200 --epochs 1 # 冒烟测试

每轮结束在验证集上算「逐 token」和「实体级」两套指标，按实体级 F1 存最优模型。
日志、配置、指标都写进 runs/<run_id>/，这是 run_info.json 的原料。
"""

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import ARTIFACTS, MAX_LEN, RUNS
from .data_io import load_split
from .dataset import build_loader, collect_examples, load_tokenizer
from .metrics import compute_metrics, decode_batch, full_report
from .model import build_model


@dataclass
class TrainConfig:
    lr: float = 3e-5
    batch_size: int = 32
    epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    dropout: float = 0.1
    seed: int = 42
    max_len: int = MAX_LEN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        print(msg)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


@torch.no_grad()
def evaluate(model, loader, examples, device):
    """在验证集上跑一遍。返回指标字典、每类报告、非法 BIO 修复处数。"""
    model.eval()
    gold, pred = [], []
    total_loss, n_batch, n_repair = 0.0, 0, 0
    idx = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        total_loss += out.loss.item()
        n_batch += 1

        pred_ids = out.logits.argmax(dim=-1).cpu()
        decoded, repaired = decode_batch(pred_ids, batch["offsets"], batch["text"])
        n_repair += repaired
        pred.extend(decoded)
        for _ in range(len(decoded)):
            gold.append(examples[idx]["tags"])
            idx += 1

    metrics = compute_metrics(gold, pred)
    metrics["val_loss"] = total_loss / max(n_batch, 1)
    metrics["bio_repairs"] = n_repair
    return metrics, full_report(gold, pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit-dialogues", type=int, default=None,
                    help="只用前 N 段对话，冒烟测试用")
    args = ap.parse_args()

    cfg = TrainConfig(lr=args.lr, batch_size=args.batch_size,
                      epochs=args.epochs, seed=args.seed)
    run_id = args.run_id or (
        f"softmax_lr{cfg.lr:g}_bs{cfg.batch_size}_ep{cfg.epochs}_seed{cfg.seed}"
        f"_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = RUNS / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / "train.log")
    log(f"run_id={run_id}")
    log(f"配置: {json.dumps(asdict(cfg), ensure_ascii=False)}")
    # 先落一份配置：这样即使训练中途崩了，实验记录表里也能看到这次用了什么参数
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "run_id": run_id, "history": []},
                  f, ensure_ascii=False, indent=2)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备: {device}  {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    with open(ARTIFACTS / "split_v1.json", encoding="utf-8") as f:
        split = json.load(f)
    train_ids, val_ids = split["train"], split["val"]
    if args.limit_dialogues:
        train_ids = train_ids[:args.limit_dialogues]
        val_ids = val_ids[:50]
        log(f"冒烟模式：train {len(train_ids)} 段对话，val {len(val_ids)} 段")

    data = load_split("train")
    tokenizer = load_tokenizer()
    train_examples = collect_examples(data, train_ids)
    val_examples = collect_examples(data, val_ids)
    log(f"训练句数 {len(train_examples)}，验证句数 {len(val_examples)}")

    train_loader = build_loader(train_examples, tokenizer, batch_size=cfg.batch_size,
                                shuffle=True, max_len=cfg.max_len)
    val_loader = build_loader(val_examples, tokenizer, batch_size=64,
                              shuffle=False, max_len=cfg.max_len)

    model = build_model(dropout=cfg.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"参数量 {n_params / 1e6:.1f}M")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}")
        running = 0.0
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / 1.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = running / max(len(train_loader), 1)
        metrics, report = evaluate(model, val_loader, val_examples, device)
        metrics.update({"epoch": epoch, "train_loss": avg_loss,
                        "elapsed_seconds": time.time() - t0})
        history.append(metrics)
        log(f"epoch {epoch} 训练损失 {avg_loss:.4f} 验证损失 {metrics['val_loss']:.4f} "
            f"token准 {metrics['token_accuracy']:.4f} token宏F1 {metrics['token_macro_f1']:.4f} "
            f"实体P {metrics['entity_precision']:.4f} R {metrics['entity_recall']:.4f} "
            f"F1 {metrics['entity_f1']:.4f}")
        log(report)

        if metrics["entity_f1"] > best_f1:
            best_f1 = metrics["entity_f1"]
            torch.save({"model_state": model.state_dict(), "config": asdict(cfg),
                        "epoch": epoch, "metrics": metrics},
                       run_dir / "checkpoints" / "best.pt")
            log(f"  已保存最优模型（实体 F1 {best_f1:.4f}）")

        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "run_id": run_id,
                   "train_sentences": len(train_examples),
                   "val_sentences": len(val_examples),
                   "history": history}, f, ensure_ascii=False, indent=2)
    log(f"完成，总耗时 {time.time() - t0:.1f} 秒，最优实体 F1 {best_f1:.4f}")
    log(f"输出目录 {run_dir}")


if __name__ == "__main__":
    main()
