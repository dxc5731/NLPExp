# -*- coding: utf-8 -*-
"""评测：逐 token 指标 + 实体级指标，两个都算，用来支撑报告里的口径对比。

为什么两个都要：
    训练集 57.6% 的句子一个实体都没有。全部预测成 O，逐 token 准确率也有 83%，
    但实体级 F1 是 0。只报准确率会得出「模型不错」的错误结论。

口径：
    逐 token  —— 每个位置算一题，11 类做分类，报准确率和 macro-F1
    实体级    —— 把连续的 B-X I-X 合成一个实体，边界和类型都对才算对（strict）
    两者都在字符级上算，和最终交接的形态一致。
"""

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score as token_f1

from .align import tokens_to_char_labels
from .labels import LABELS


def decode_batch(pred_ids, offsets, texts, id2label_ids=None):
    """词片级预测 -> 每个字符的标签字符串列表。"""
    out = []
    for i, text in enumerate(texts):
        n = len(text)
        offs = offsets[i].tolist()
        # 只取覆盖了字符的词片
        pairs = [(int(p), tuple(o)) for p, o in zip(pred_ids[i].tolist(), offs) if o[0] != o[1]]
        labels = [int(p) for p, _ in pairs]
        o = [o for _, o in pairs]
        tags, _ = tokens_to_char_labels(labels, o, n)
        out.append(tags)
    return out


def compute_metrics(gold_tags, pred_tags):
    """gold_tags / pred_tags 都是「每句一个标签列表」。"""
    flat_gold = [t for seq in gold_tags for t in seq]
    flat_pred = [t for seq in pred_tags for t in seq]
    token_acc = accuracy_score(flat_gold, flat_pred)
    token_macro = token_f1(flat_gold, flat_pred, average="macro", zero_division=0, labels=LABELS)
    return {
        "token_accuracy": token_acc,
        "token_macro_f1": token_macro,
        "entity_precision": precision_score(gold_tags, pred_tags, zero_division=0),
        "entity_recall": recall_score(gold_tags, pred_tags, zero_division=0),
        "entity_f1": f1_score(gold_tags, pred_tags, zero_division=0),
    }


def full_report(gold_tags, pred_tags) -> str:
    return classification_report(gold_tags, pred_tags, digits=4, zero_division=0)
