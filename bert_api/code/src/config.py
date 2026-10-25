# -*- coding: utf-8 -*-
"""全局路径与常量。所有模块都从这里拿路径，不要在别处写死。

数据目录与模型目录都支持用环境变量覆盖（放到别的盘时用）：
    set IMCS_DATA_DIR=D:\\datasets\\IMCS-V2-NER
    set BERT_MODEL_DIR=D:\\models\\bert-base-chinese

没设环境变量时，自动在几个常见位置找数据，找到哪个用哪个：
    1. <项目根>/IMCS-V2-NER                     ← 本作业实际存放处
    2. <项目根>/IMCS-V2-NER/IMCS-V2-NER         ← 解压时多套一层
    3. <bert_api>/doc/数据/IMCS-V2-NER/IMCS-V2-NER  ← 原始交接包默认路径
这样老师 clone 下来把数据放到项目根 IMCS-V2-NER/ 即可，无需设环境变量。
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # bert_api/
CODE = ROOT / "code"
PROJECT_ROOT = ROOT.parent                       # 作业根目录（baseline/ 与 bert_api/ 的上一层）

_here = Path(__file__).resolve()


def _pick_data_dir() -> Path:
    """按优先级挑一个真实存在的数据目录；都不存在时退回项目根下的 IMCS-V2-NER。"""
    env = os.environ.get("IMCS_DATA_DIR")
    if env:
        return Path(env)
    candidates = [
        PROJECT_ROOT / "IMCS-V2-NER",
        PROJECT_ROOT / "IMCS-V2-NER" / "IMCS-V2-NER",
        ROOT / "doc" / "数据" / "IMCS-V2-NER" / "IMCS-V2-NER",
    ]
    for c in candidates:
        if (c / "IMCS-V2_train.json").exists():
            return c
    return candidates[0]


DATA_DIR = _pick_data_dir()

_default_model = CODE / "models" / "bert-base-chinese"
MODEL_DIR = Path(os.environ.get("BERT_MODEL_DIR") or _default_model)

ARTIFACTS = CODE / "artifacts"
RUNS = PROJECT_ROOT / "runs"

# 最长句 199 字符，加上 [CLS]/[SEP] 也远小于 256，不需要滑窗。
MAX_LEN = 256
