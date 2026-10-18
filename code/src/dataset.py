# -*- coding: utf-8 -*-
"""统一的 PyTorch 数据接口。BERT-Softmax 和 BERT-CRF 都用这一份，保证对比公平。

一个 batch 里给你四样东西：

    input_ids        [B, L]  词片编号
    attention_mask   [B, L]  1=真实词片，0=补位
    labels           [B, L]  词片级标签；特殊符号和补位是 -100，损失函数会自动跳过
    offsets          [B, L, 2] 每个词片对应原文的 [起, 止) 字符下标，解码还原时用

为什么 labels 用 -100 而不是 PAD 标签：
    PyTorch 的交叉熵默认忽略 -100。这样补位和 [CLS]/[SEP] 不参与训练，
    也不用担心模型把补位学成 O，把 F1 刷虚高。

为什么把 offsets 也放进 batch：
    推理时要把词片预测还原成字符预测，必须知道每个词片覆盖哪几个字。
    训练时不用，但留着一份，接口对两边一致，B 那边写 CRF 时不用改结构。
"""

import torch
from torch.utils.data import DataLoader, Dataset

from .align import IGNORE_INDEX, encode_sentence, gold_to_token_labels, load_tokenizer
from .config import MAX_LEN


class NerDataset(Dataset):
    """把「对话ID + 句子ID + 原文 + 标签」变成模型输入。

    分词只在初始化时做一次，训练时直接取，不重复算。
    """

    def __init__(self, examples, tokenizer, max_len: int = MAX_LEN, with_labels: bool = True):
        self.examples = examples
        self.with_labels = with_labels
        self.encoded = []
        for ex in examples:
            ids, offsets = encode_sentence(tokenizer, ex["text"], max_len=max_len)
            item = {"input_ids": ids, "offsets": offsets}
            if with_labels:
                item["labels"] = gold_to_token_labels(ex["tags"], offsets)
            self.encoded.append(item)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        ex, enc = self.examples[i], self.encoded[i]
        out = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.ones(len(enc["input_ids"]), dtype=torch.long),
            "offsets": torch.tensor(enc["offsets"], dtype=torch.long),
            "dialogue_id": ex["dialogue_id"],
            "sentence_id": ex["sentence_id"],
            "text": ex["text"],
        }
        if self.with_labels:
            out["labels"] = torch.tensor(enc["labels"], dtype=torch.long)
        return out


def collate_fn(batch, pad_token_id: int = 0):
    """把一个 batch 里的句子补到等长。补位标签填 -100。"""
    max_len = max(len(item["input_ids"]) for item in batch)
    b = len(batch)

    input_ids = torch.full((b, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((b, max_len), dtype=torch.long)
    offsets = torch.zeros((b, max_len, 2), dtype=torch.long)
    labels = torch.full((b, max_len), IGNORE_INDEX, dtype=torch.long)

    for i, item in enumerate(batch):
        ids = torch.as_tensor(item["input_ids"], dtype=torch.long)
        n = len(ids)
        input_ids[i, :n] = ids
        attention_mask[i, :n] = torch.as_tensor(item["attention_mask"], dtype=torch.long)
        offsets[i, :n] = torch.as_tensor(item["offsets"], dtype=torch.long)
        if "labels" in item:
            labels[i, :n] = torch.as_tensor(item["labels"], dtype=torch.long)

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offsets": offsets,
        "dialogue_id": [item["dialogue_id"] for item in batch],
        "sentence_id": [item["sentence_id"] for item in batch],
        "text": [item["text"] for item in batch],
    }
    if "labels" in batch[0]:
        out["labels"] = labels
    return out


def collect_examples(data: dict, dialogue_ids=None, require_labels: bool = True):
    """从原始 JSON 里抽出句子清单。dialogue_ids 传 None 表示全部。"""
    ids = list(data.keys()) if dialogue_ids is None else list(dialogue_ids)
    examples = []
    for did in ids:
        for s in data[did]["dialogue"]:
            text = s["sentence"]
            tags = s["BIO_label"].split(" ") if require_labels else None
            if require_labels:
                assert len(tags) == len(text), (did, s["sentence_id"])
            examples.append({
                "dialogue_id": did,
                "sentence_id": s["sentence_id"],
                "text": text,
                "tags": tags,
            })
    return examples


def build_loader(examples, tokenizer, batch_size: int = 32, shuffle: bool = False,
                 max_len: int = MAX_LEN, with_labels: bool = True, num_workers: int = 0):
    """一行拿到 DataLoader。B 那边直接用这个。"""
    ds = NerDataset(examples, tokenizer, max_len=max_len, with_labels=with_labels)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        num_workers=num_workers,
    )


__all__ = [
    "NerDataset", "collate_fn", "collect_examples", "build_loader",
    "load_tokenizer", "MAX_LEN", "IGNORE_INDEX",
]
