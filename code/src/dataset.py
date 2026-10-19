# -*- coding: utf-8 -*-
"""统一的 PyTorch 数据接口。BERT-Softmax 和 BERT-CRF 都用这一份，保证对比公平。

一个 batch 里给你：

    input_ids        [B, L]  词片编号
    attention_mask   [B, L]  1=真实词片，0=补位
    labels           [B, L]  词片级标签；特殊符号、补位、**无法无损表示的词片**都是 -100
    offsets          [B, L, 2] 每个词片对应原文的 [起, 止) 字符下标

除了 batch，Dataset 本身还挂着两样字符级的东西（CRF 和校验要用）：
    ds.char_labels[i]   第 i 句的字符级金标，原样，长度 = len(原文)
    ds.ambiguous        无法无损映射的词片清单（对话ID, 句子ID, 位置, 两边的标签）
    ds.report()         上面这些东西的汇总统计

为什么 -100 而不是别的：
    交叉熵默认忽略 -100。补位和特殊符号不该参与训练；
    一个词片覆盖多个字符且标签互相冲突时，任何单一标签都是伪造，
    也只能不参与训练——但会记进 ambiguous，绝不静默处理。
"""

import torch
from torch.utils.data import DataLoader, Dataset

from .align import IGNORE_INDEX, encode_sentence, load_tokenizer
from .config import MAX_LEN


class NerDataset(Dataset):
    """把「对话ID + 句子ID + 原文 + 标签」变成模型输入。分词只在初始化时做一次。"""

    def __init__(self, examples, tokenizer, max_len: int = MAX_LEN, with_labels: bool = True):
        self.examples = examples
        self.with_labels = with_labels
        self.encoded = []
        self.char_labels = [] if with_labels else None
        self.ambiguous = []          # [(dialogue_id, sentence_id, {start,end,text,tags})]
        self.n_ambiguous_sentences = 0

        for ex in examples:
            where = f"对话{ex['dialogue_id']}/句{ex['sentence_id']}"
            enc = encode_sentence(
                tokenizer, ex["text"], max_len=max_len,
                tags=ex["tags"] if with_labels else None, where=where,
            )
            item = {"input_ids": enc.input_ids, "offsets": enc.offsets}
            if with_labels:
                item["labels"] = enc.token_labels
                self.char_labels.append(list(ex["tags"]))
                if enc.ambiguous:
                    self.n_ambiguous_sentences += 1
                    for a in enc.ambiguous:
                        self.ambiguous.append((ex["dialogue_id"], ex["sentence_id"], a))
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

    def report(self) -> dict:
        return {
            "examples": len(self.examples),
            "ambiguous_tokens": len(self.ambiguous),
            "ambiguous_sentences": self.n_ambiguous_sentences,
            "ambiguous_examples": [
                {"dialogue_id": d, "sentence_id": s, **a}
                for d, s, a in self.ambiguous[:20]
            ],
        }


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
            if require_labels and len(tags) != len(text):
                raise ValueError(f"对话{did}/句{s['sentence_id']} 标签数与字符数不一致")
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
