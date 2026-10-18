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
