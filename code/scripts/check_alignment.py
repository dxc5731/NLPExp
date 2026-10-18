# -*- coding: utf-8 -*-
"""全量对齐体检：每一句都走一遍「字符 -> 词片 -> 字符」，证明没有丢东西。

检查五件事：
  1. 每个原始字符都被恰好一个词片覆盖（丢字会让标签无法还原）
  2. 有没有句子被截断（本数据集最长 199 字符，理论上不该发生）
  3. 一个词片覆盖多个字符的情况有多少、都是些什么
  4. 把金标准标签「降到词片再摊回来」，和原始标签逐字比对
  5. 词表里没有的字符（[UNK]）有多少

用法（在 code/ 目录下）：
    python -m scripts.check_alignment           # 训练集（默认）
    python -m scripts.check_alignment dev
    python -m scripts.check_alignment test      # 无金标准，只查覆盖
"""

import json
import sys
from collections import Counter

from src.align import (
    encode_sentence,
    gold_to_token_labels,
    load_tokenizer,
    tokens_to_char_labels,
)
from src.config import ARTIFACTS, MAX_LEN
from src.data_io import load_split

SHOW = 5


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    has_labels = split != "test"
    tokenizer = load_tokenizer()
    data = load_split(split)

    n_sent = 0
    n_char = 0
    total_uncovered = 0
    over_covered = 0
    uncovered_examples = []
    multi_char = Counter()
    multi_char_examples = []
    roundtrip_bad = []
    n_roundtrip_bad = 0
    truncated = 0
    max_tokens = 0
    unk = 0
    overlap = 0

    for did, d in data.items():
        for s in d["dialogue"]:
            text = s["sentence"]
            tags = s["BIO_label"].split(" ") if has_labels else None
            if has_labels:
                assert len(tags) == len(text), (did, s["sentence_id"])

            ids, offsets = encode_sentence(tokenizer, text)
            covered = [0] * len(text)
            n_sent += 1
            n_char += len(text)
            max_tokens = max(max_tokens, len(ids))
            if len(ids) >= MAX_LEN:
                truncated += 1
            unk += sum(1 for t in ids if t == tokenizer.unk_token_id)

            prev_end = 0
            for start, end in offsets:
                if start == end:          # 特殊符号是 0 长度
                    continue
                if start < prev_end:
                    overlap += 1
                prev_end = max(prev_end, end)
                if end - start > 1:
                    multi_char[end - start] += 1
                    if len(multi_char_examples) < SHOW:
                        multi_char_examples.append(
                            (did, s["sentence_id"], text[start:end], start, end))
                for k in range(start, end):
                    covered[k] += 1

            missed = [k for k in range(len(text)) if covered[k] == 0]
            total_uncovered += len(missed)
            over_covered += sum(1 for v in covered if v > 1)
            if missed and len(uncovered_examples) < SHOW:
                uncovered_examples.append(
                    (did, s["sentence_id"], missed[:5], [repr(text[k]) for k in missed[:5]]))

            if has_labels:
                token_labels = gold_to_token_labels(tags, offsets)
                restored, _ = tokens_to_char_labels(token_labels, offsets, len(text))
                if restored != tags:
                    n_roundtrip_bad += 1
                    if len(roundtrip_bad) < SHOW:
                        diff = [i for i, (a, b) in enumerate(zip(tags, restored)) if a != b]
                        roundtrip_bad.append((did, s["sentence_id"], text, diff[:5],
                                              [tags[i] for i in diff[:5]],
                                              [restored[i] for i in diff[:5]]))

    print(f"=== {split} ===")
    print(f"句子数 {n_sent}  字符数 {n_char}")
    print(f"最长输入长度 {max_tokens}（上限 {MAX_LEN}），被截断句子 {truncated}")
    print(f"未被覆盖的字符 {total_uncovered}，重复覆盖 {over_covered}，offset 重叠 {overlap}")
    print(f"[UNK] 词片 {unk}")
    print(f"一个词片覆盖多字符: {dict(sorted(multi_char.items()))}")
    for e in multi_char_examples:
        print("   例:", e)
    if has_labels:
        print(f"金标准往返不一致的句子 {n_roundtrip_bad} ({n_roundtrip_bad / max(n_sent, 1) * 100:.4f}%)")
        for r in roundtrip_bad:
            print("   例:", r)
    if uncovered_examples:
        print("未被覆盖的字符样例:", uncovered_examples)

    report = {
        "split": split,
        "sentences": n_sent,
        "chars": n_char,
        "max_tokens": max_tokens,
        "truncated": truncated,
        "uncovered_chars": total_uncovered,
        "over_covered_chars": over_covered,
        "offset_overlap": overlap,
        "unk_tokens": unk,
        "multi_char_tokens": {str(k): v for k, v in sorted(multi_char.items())},
        "roundtrip_mismatch_sentences": n_roundtrip_bad,
        "roundtrip_mismatch_ratio": n_roundtrip_bad / max(n_sent, 1),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / f"alignment_report_{split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告写入", out)

    assert truncated == 0, "有句子被截断"
    assert total_uncovered == 0, "有字符没被任何词片覆盖"
    assert overlap == 0, "offset 有重叠"
    # 词片比字符粗，极少数位置会把相邻的两个标签合并。这是分词本身的信息损失，
    # 不是代码 bug。用比例卡住，别让它悄悄涨上去。
    assert n_roundtrip_bad / max(n_sent, 1) < 0.001, "金标准往返不一致比例过高"
    print("全部检查通过")


if __name__ == "__main__":
    main()
