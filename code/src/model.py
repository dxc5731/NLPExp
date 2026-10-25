# -*- coding: utf-8 -*-
"""BERT + Softmax 序列标注模型。

结构很简单：预训练 BERT 出每个词片的向量，接一层 11 维的线性分类器，
每个位置独立取 argmax。没有 CRF，所以可能出现非法标号序列（比如句首 I-X），
我们在解码那一步会修复——这一点正好是后面和 BERT-CRF 对比的落点。
"""

from transformers import AutoModelForTokenClassification

from .config import MODEL_DIR, require_model_dir
from .labels import ID2LABEL, LABEL2ID, LABELS


def build_model(dropout: float = 0.1, local_files_only: bool = True):
    require_model_dir()
    model = AutoModelForTokenClassification.from_pretrained(
        str(MODEL_DIR),
        num_labels=len(LABELS),
        id2label=dict(ID2LABEL),
        label2id=dict(LABEL2ID),
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        local_files_only=local_files_only,
    )
    return model
