# -*- coding: utf-8 -*-
"""接口自测：假装我是 B，只拿接口，看能不能直接训。

跑完你会看到：
  1. 一个 batch 的张量形状
  2. 第一句的「原文 / 词片 / 词片标签」
  3. 把词片标签摊回字符后，和原始金标准逐字对比

用法（在 code/ 目录下）：
    python -m scripts.demo_dataloader
"""

import json

from src.align import tokens_to_char_labels
from src.config import ARTIFACTS
from src.data_io import load_split
from src.dataset import build_loader, collect_examples, load_tokenizer
from src.labels import ID2LABEL

N_DIALOGUES = 200
BATCH_SIZE = 4


def main() -> None:
    tokenizer = load_tokenizer()

    with open(ARTIFACTS / "split_v1.json", encoding="utf-8") as f:
        split = json.load(f)
    train_ids = split["train"][:N_DIALOGUES]

    data = load_split("train")
    examples = collect_examples(data, train_ids)
    loader = build_loader(examples, tokenizer, batch_size=BATCH_SIZE, shuffle=False)
    print(f"用了 {len(train_ids)} 段对话，共 {len(examples)} 句")

    batch = next(iter(loader))
    print("\n=== batch 形状 ===")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  {k:16s} {tuple(v.shape)}  {v.dtype}")

    i = 0
    text = batch["text"][i]
    offsets = batch["offsets"][i]
    labels = batch["labels"][i]
    ids = batch["input_ids"][i]
    mask = batch["attention_mask"][i]

    n_real = int(mask.sum())
    tokens = tokenizer.convert_ids_to_tokens(ids[:n_real].tolist())
    print("\n=== 第一句 ===")
    print("原文:", text)
    print("词片:", tokens)
    print("逐片标签:", [ID2LABEL[int(t)] if int(t) != -100 else "_" for t in labels[:n_real]])

    # 把词片标签摊回字符，和原始金标准比
    restored, uncovered = tokens_to_char_labels(labels[:n_real].tolist(), offsets[:n_real].tolist(), len(text))
    gold = data[batch["dialogue_id"][i]]["dialogue"]
    gold_tags = None
    for s in gold:
        if s["sentence_id"] == batch["sentence_id"][i]:
            gold_tags = s["BIO_label"].split(" ")
    print("还原后:", " ".join(restored))
    print("金标准:", " ".join(gold_tags))
    print("一致:" , restored == gold_tags, " 未覆盖字符:", uncovered)


if __name__ == "__main__":
    main()
