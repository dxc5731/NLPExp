# -*- coding: utf-8 -*-
"""全局路径与常量。所有模块都从这里拿路径，不要在别处写死。

数据目录和模型目录可以用环境变量覆盖，方便队友把它放到别的盘：
    set IMCS_DATA_DIR=D:\\datasets\\IMCS-V2-NER\\IMCS-V2-NER
    set BERT_MODEL_DIR=D:\\models\\bert-base-chinese
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # 项目根 nlp/
CODE = ROOT / "code"

_default_data = ROOT / "doc" / "数据" / "IMCS-V2-NER" / "IMCS-V2-NER"
DATA_DIR = Path(os.environ.get("IMCS_DATA_DIR") or _default_data)

_default_model = CODE / "models" / "bert-base-chinese"
MODEL_DIR = Path(os.environ.get("BERT_MODEL_DIR") or _default_model)

ARTIFACTS = CODE / "artifacts"
RUNS = ROOT / "runs"

# 最长句 199 字符，加上 [CLS]/[SEP] 也远小于 256，不需要滑窗。
MAX_LEN = 256


def require_model_dir() -> Path:
    """模型目录不存在时给一句人能看懂的话，而不是让 transformers 抛奇怪的报错。"""
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"找不到模型目录：{MODEL_DIR}\n"
            f"请下载 bert-base-chinese 放到该目录，或设置环境变量 BERT_MODEL_DIR 指向已下载的模型目录。\n"
            f"例如（PowerShell）：$env:BERT_MODEL_DIR='D:\\models\\bert-base-chinese'"
        )
    return MODEL_DIR


def require_data_dir() -> Path:
    """数据目录不存在时同样给明确提示。"""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"找不到数据目录：{DATA_DIR}\n"
            f"请把 IMCS-V2-NER 数据集放到该目录，或设置环境变量 IMCS_DATA_DIR 指向含 JSON 的目录。\n"
            f"例如（PowerShell）：$env:IMCS_DATA_DIR='D:\\datasets\\IMCS-V2-NER\\IMCS-V2-NER'"
        )
    return DATA_DIR
