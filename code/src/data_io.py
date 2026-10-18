# -*- coding: utf-8 -*-
"""读原始数据的最小工具层。

约定：所有路径都从项目根目录推出来，不写死绝对路径。
    code/src/data_io.py  ->  parents[0]=src, [1]=code, [2]=项目根
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "doc" / "数据" / "IMCS-V2-NER" / "IMCS-V2-NER"

SPLITS = {
    "train": "IMCS-V2_train.json",
    "dev": "IMCS-V2_dev.json",
    "test": "IMCS-V2_test.json",
}


def load_split(split: str) -> dict:
    """读一个官方划分，返回 {对话ID: 对话字典}。"""
    path = DATA_DIR / SPLITS[split]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    """算文件字节的 SHA-256。run_info 里要填，先算出来备用。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
