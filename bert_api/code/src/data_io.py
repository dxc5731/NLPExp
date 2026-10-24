# -*- coding: utf-8 -*-
"""读原始数据的最小工具层。"""

import hashlib
import json

from .config import DATA_DIR

SPLITS = {
    "train": "IMCS-V2_train.json",
    "dev": "IMCS-V2_dev.json",
    "test": "IMCS-V2_test.json",
}


def split_path(split: str):
    return DATA_DIR / SPLITS[split]


def load_split(split: str) -> dict:
    """读一个官方划分，返回 {对话ID: 对话字典}。"""
    with open(split_path(split), "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path) -> str:
    """算文件字节的 SHA-256。run_info 里要填，先算出来备用。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
