# -*- coding: utf-8 -*-
"""接口自测：假装我是 B，只拿接口，看能不能直接训。

跑完你会看到：
  1. 一个 batch 的张量形状
  2. 第一句的「原文 / 词片 / 词片标签」
  3. 把词片标签摊回字符后，和原始金标准逐字对比
  4. 这批数据里有没有无法无损映射的词片

用法（在 code/ 目录下）：
    python -m scripts.demo_dataloader
"""

import json

from src.align import restore_char_labels
from src.config import ARTIFACTS
from src.data_io import load_split
from src.dataset import NerDataset, build_loader, collect_examples, load_tokenizer
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

    ds = NerDataset(examples, tokenizer)
    rep = ds.report()
    print(f"用了 {len(train_ids)} 段对话，共 {rep['examples']} 句")
    print(f"无法无损映射的词片: {rep['ambiguous_tokens']} 个，涉及 {rep['ambiguous_sentences']} 句")
    for a in rep["ambiguous_examples"][:3]:
        print("   例:", a)

    loader = build_loader(examples, tokenizer, batch_size=BATCH_SIZE, shuffle=False)
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

    restored = restore_char_labels(labels[:n_real].tolist(), offsets[:n_real].tolist(), len(text))
    gold = ds.char_labels[i]
    print("还原后:", " ".join(t if t else "<无标签>" for t in restored))
    print("金标准:", " ".join(gold))
    print("一致:", restored == gold)


if __name__ == "__main__":
    main()
