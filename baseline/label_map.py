# -*- coding: utf-8 -*-
"""BIO 合法性修复。标签表不再本地复制，统一从 src/labels.py 导入。

为什么改这个：
    原来这里自己抄了一份 11 个标签（和 src/labels.py 内容重复）。
    `src/labels.py` 是 C 定的全项目唯一标签来源——模型输出层的第 i 个神经元
    对应它的第 i 个标签。一旦两处顺序不一致，模型预测和基线打分的标签就会错位，
    整张对比表静默变错（不报错，只是数字不对）。
    这里只保留 `check_bio` 这个纯逻辑函数，标签列表改为 import，杜绝漂移。

依赖说明：src/labels.py 是纯 Python（无 torch/transformers），导入它很轻，
不会给基线带来额外负担。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(os.path.dirname(_HERE), "bert_api", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.labels import LABEL2ID as label2id  # noqa: E402
from src.labels import LABELS  # noqa: E402

id2label = {i: l for l, i in label2id.items()}


def check_bio(tags):
    """合法性修复：I 开头改 B；类型跳变截断为 B。tags: List[str] -> List[str]"""
    out = []
    prev_type = None
    for t in tags:
        if t == "O":
            out.append(t); prev_type = None; continue
        if "-" not in t:
            out.append("O"); prev_type = None; continue
        pre, typ = t.split("-", 1)
        if pre not in ("B", "I") or t not in label2id:
            out.append("O"); prev_type = None; continue
        if pre == "I" and prev_type != typ:
            out.append("B-" + typ); prev_type = typ
        else:
            out.append(t); prev_type = typ
    return out
