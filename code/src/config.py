# -*- coding: utf-8 -*-
"""全局路径与常量。所有模块都从这里拿路径，不要在别处写死。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # 项目根 nlp/
CODE = ROOT / "code"

DATA_DIR = ROOT / "doc" / "数据" / "IMCS-V2-NER" / "IMCS-V2-NER"
MODEL_DIR = CODE / "models" / "bert-base-chinese"
ARTIFACTS = CODE / "artifacts"
RUNS = ROOT / "runs"

# 最长句 199 字符，加上 [CLS]/[SEP] 也远小于 256，不需要滑窗。
MAX_LEN = 256
