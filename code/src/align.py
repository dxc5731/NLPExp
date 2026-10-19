# -*- coding: utf-8 -*-
"""字符级 BIO 与 BERT 词片的对齐层（v2）。

v1 的问题（成员 B 指出，已确认）：
    v1 在「一个词片覆盖多个字符」时，直接取首字符标签当整个词片的标签；
    还原时又自动把非法的 I 改成 B、把没覆盖到的字符补成 O。
    这两处都会**静默伪造标签**，把对齐错误盖住。

v2 的规则：
    1. 字符级金标是唯一真相，原样保留，不参与任何改写。
    2. 一个词片的标签，必须是「能把该词片覆盖的每个字符都逐字还原出来」的那个标签。
       还原规则固定：首字符用原标签，后续字符若原标签是 B-X 就降级成 I-X。
       一个标签做不到逐字还原，这个位置就**不参与训练（-100）**，
       并记录到 ambiguous 里，附位置和两边的标签，绝不猜一个顶上。
    3. 还原函数不做任何修复：没覆盖到的字符返回 None，非法的 I 原样保留。
       要合法 BIO 请显式调用 to_legal_bio()，它会返回改了几处。
    4. 超过长度上限时直接报错，不静默截断。
"""

from dataclasses import dataclass, field

from .config import MAX_LEN
from .labels import LABEL2ID, LABELS

IGNORE_INDEX = -100


class AlignmentError(Exception):
    """对齐失败。带对话 ID / 句子 ID / 字符位置，方便直接定位。"""


@dataclass
class Encoded:
    input_ids: list
    offsets: list            # [(start, end), ...]，特殊符号是 (0, 0)
    token_labels: list = None
    ambiguous: list = field(default_factory=list)
    n_char: int = 0

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)


def load_tokenizer():
    from transformers import AutoTokenizer

    from .config import MODEL_DIR

    return AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)


def _expand(tag: str, k: int, start: int) -> str:
    """词片标签摊到它覆盖的第 k 个字符上。"""
    if k == start:
        return tag
    return "I-" + tag[2:] if tag.startswith("B-") else tag


def encode_sentence(tokenizer, text: str, max_len: int = MAX_LEN,
                    tags=None, where: str = "") -> Encoded:
    """一句话 -> 词片编号 + 字符区间 + 词片标签（可选）。

    tags 给定时同时算 token_labels；无法无损表示的词片填 -100 并记进 ambiguous。
    """
    if tags is not None and len(tags) != len(text):
        raise AlignmentError(f"{where} 标签数 {len(tags)} 与字符数 {len(text)} 不一致")

    enc = tokenizer(text, add_special_tokens=True, truncation=False,
                    return_offsets_mapping=True)
    input_ids = list(enc["input_ids"])
    offsets = [tuple(int(x) for x in o) for o in enc["offset_mapping"]]

    if len(input_ids) > max_len:
        raise AlignmentError(
            f"{where} 分词后 {len(input_ids)} 个词片，超过上限 {max_len}；"
            f"拒绝静默截断（原文 {len(text)} 字符）")

    # 覆盖检查：每个字符必须恰好被一个词片覆盖
    covered = [0] * len(text)
    prev_end = 0
    for start, end in offsets:
        if start == end:
            continue
        if start < prev_end:
            raise AlignmentError(f"{where} 词片区间重叠：{start} < {prev_end}")
        if end > len(text):
            raise AlignmentError(f"{where} 词片区间越界：{end} > {len(text)}")
        prev_end = end
        for k in range(start, end):
            covered[k] += 1
    uncovered = [k for k, v in enumerate(covered) if v == 0]
    if uncovered:
        raise AlignmentError(
            f"{where} 有 {len(uncovered)} 个字符没有被任何词片覆盖，位置 {uncovered[:10]}")

    token_labels = None
    ambiguous = []
    if tags is not None:
        token_labels = []
        for start, end in offsets:
            if start == end:
                token_labels.append(IGNORE_INDEX)
                continue
            tag0 = tags[start]
            if all(tags[k] == _expand(tag0, k, start) for k in range(start, end)):
                token_labels.append(LABEL2ID[tag0])
            else:
                token_labels.append(IGNORE_INDEX)
                ambiguous.append({
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "tags": [tags[k] for k in range(start, end)],
                })

    return Encoded(input_ids=input_ids, offsets=offsets, token_labels=token_labels,
                   ambiguous=ambiguous, n_char=len(text))


def restore_char_labels(token_label_ids, offsets, n_char: int):
    """词片级标签 -> 字符级标签。**不做任何修复**。

    返回长度 n_char 的列表，元素是标签字符串或 None。
    None 表示：该字符所在词片没有标签（特殊符号、被排除的歧义词片），
    或者根本没有词片覆盖它。调用方必须自己决定怎么处理，不能当作 O。
    """
    out = [None] * n_char
    for label_id, (start, end) in zip(token_label_ids, offsets):
        if start == end or int(label_id) == IGNORE_INDEX:
            continue
        tag = LABELS[int(label_id)]
        for k in range(start, end):
            out[k] = _expand(tag, k, start)
    return out


def to_legal_bio(tags):
    """把预测结果修成合法 BIO。**只用于导出预测**，不用于对齐校验。

    返回 (修复后的标签, 修复处数)。每一处修改都是显式的，调用方应记日志。
    """
    fixed, n_changed = [], 0
    for tag in tags:
        if tag is None:
            raise AlignmentError("存在没有标签的字符，不能直接导出")
        if tag.startswith("I-"):
            ent = tag[2:]
            prev = fixed[-1] if fixed else None
            if prev not in ("B-" + ent, "I-" + ent):
                tag = "B-" + ent
                n_changed += 1
        fixed.append(tag)
    return fixed, n_changed
