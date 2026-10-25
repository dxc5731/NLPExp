# -*- coding: utf-8 -*-
"""服务器自检：一条命令看清这台机器能不能跑，不能跑卡在哪一步。

用法（在 code/ 目录下）：
    python -X utf8 -m scripts.doctor

任何一步失败都会直接抛错，报错信息就是需要发回来的内容。
"""

import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, MODEL_DIR, ROOT  # noqa: E402
from src.data_io import SPLITS  # noqa: E402


def line(msg=""):
    print(msg, flush=True)


def main() -> None:
    line("=" * 60)
    line("1. 解释器")
    line(f"   Python {sys.version.split()[0]}  {platform.platform()}")
    line(f"   项目根 {ROOT}")
    line(f"   数据目录 {DATA_DIR}  {'OK' if DATA_DIR.exists() else '不存在'}")
    line(f"   模型目录 {MODEL_DIR}  {'OK' if MODEL_DIR.exists() else '不存在'}")

    line("\n2. 数据文件")
    for split, name in SPLITS.items():
        p = DATA_DIR / name
        if p.exists():
            line(f"   {name:24s} {p.stat().st_size / 1e6:7.1f} MB")
        else:
            line(f"   {name:24s} 缺失")

    line("\n3. PyTorch / GPU")
    import torch

    line(f"   torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        line(f"   GPU {prop.name}  {prop.total_memory / 1e9:.1f} GB  "
             f"compute {prop.major}.{prop.minor}")
        line(f"   CUDA {torch.version.cuda}  cudnn {torch.backends.cudnn.version()}")

    line("\n4. transformers")
    import transformers

    line(f"   transformers {transformers.__version__}")

    line("\n5. 走一遍真实数据：分词 -> 模型前向+反向")
    from src.data_io import load_split
    from src.dataset import build_loader, collect_examples, load_tokenizer
    from src.model import build_model

    data = load_split("train")
    line(f"   train 对话数 {len(data)}")
    tok = load_tokenizer()
    line(f"   词表大小 {tok.vocab_size}")

    examples = collect_examples(data, list(data)[:20])
    loader = build_loader(examples, tok, batch_size=8)
    batch = next(iter(loader))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    model.train()
    t0 = time.time()
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )
    out.loss.backward()
    dt = time.time() - t0
    line(f"   batch 形状 {tuple(batch['input_ids'].shape)}")
    line(f"   loss {out.loss.item():.4f}  前向+反向 {dt * 1000:.0f} ms")

    line("\n全部自检通过，可以开始训练。")
    line("=" * 60)


if __name__ == "__main__":
    main()
