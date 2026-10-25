# -*- coding: utf-8 -*-
"""把「字符级 BIO」和「BERT 词片」对齐，并能双向换算。

三个核心动作：

1. encode_sentence   一句话 -> input_ids + 每个词片对应原文的 [起, 止) 字符区间
2. gold_to_token_labels  字符级金标准 -> 词片级标签（特殊符号和词片续段填 -100）
3. tokens_to_char_labels 词片级预测 -> 字符级标签（交接要求的形态）

为什么第 3 步不能简单复制：
    一个词片可能覆盖多个字符（实测 "38" 是一个词片，对应 '3' 和 '8' 两个字）。
    直接复制会让 "38" 里的 '8' 也拿到 B-，凭空把实体截断。
    规则：一个词片覆盖多个字符时，第一个字符拿原标签，后面的字符
          如果原标签是 B-X 就降级成 I-X，其余照抄。
    再补一条：句首或者 O 后面出现的 I-X 一律改写成 B-X，保证 BIO 合法。
"""

from .config import MAX_LEN
from .labels import LABEL2ID, LABELS

IGNORE_INDEX = -100


def load_tokenizer():
    """从项目内的本地目录加载，不联网。"""
    from transformers import AutoTokenizer

    from .config import MODEL_DIR

    return AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)


def encode_sentence(tokenizer, text: str, max_len: int = MAX_LEN):
    """返回 (input_ids, offset_mapping)。offset 是 [起, 止) 的字符下标。"""
    enc = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
    )
    return enc["input_ids"], enc["offset_mapping"]


def gold_to_token_labels(char_tags, offset_mapping):
    """字符级金标准 -> 词片级标签。特殊符号和覆盖 0 个字符的位置填 -100。"""
    labels = []
    for start, end in offset_mapping:
        if start == end:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(LABEL2ID[char_tags[start]])
    return labels


def tokens_to_char_labels(token_label_ids, offset_mapping, n_char: int):
    """词片级标签 -> 字符级标签。返回 (标签字符串列表, 未被覆盖的字符下标列表)。"""
    out = [None] * n_char
    for label_id, (start, end) in zip(token_label_ids, offset_mapping):
        if start == end:
            continue
        tag = LABELS[label_id]
        for k in range(start, end):
            if k == start:
                out[k] = tag
            elif tag.startswith("B-"):
                out[k] = "I-" + tag[2:]
            else:
                out[k] = tag

    uncovered = [k for k, v in enumerate(out) if v is None]
    for k in uncovered:
        out[k] = "O"

    # 修复 BIO：I-X 前面必须是同句的 B-X 或 I-X
    fixed = []
    for tag in out:
        if tag.startswith("I-"):
            ent = tag[2:]
            prev = fixed[-1] if fixed else None
            if prev not in ("B-" + ent, "I-" + ent):
                tag = "B-" + ent
        fixed.append(tag)
    return fixed, uncovered
