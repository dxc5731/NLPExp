# -*- coding: utf-8 -*-
"""全量对齐体检（v2 语义）。

检查四件事：
  1. 每个原始字符都被恰好一个词片覆盖；有没被覆盖的就直接报错（不补 O）
  2. 有没有句子超长；超长直接报错（不截断）
  3. 把金标降到词片再摊回字符，和原始金标逐字比对
  4. 哪些词片是「一个标签还原不出全部字符」的，全部列出来

v2 的关键区别：第 4 类不再猜测标签，而是标成 -100 不参与训练，并在这里明确列出。
所以通过标准是「所有非歧义句子逐字一致 + 歧义清单可追溯」。

用法（在 code/ 目录下）：
    python -m scripts.check_alignment           # train
    python -m scripts.check_alignment dev
    python -m scripts.check_alignment test
"""

import json
import sys

from src.align import AlignmentError, encode_sentence, load_tokenizer, restore_char_labels
from src.config import ARTIFACTS, MAX_LEN
from src.data_io import load_split

SHOW = 20


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    has_labels = split != "test"
    tokenizer = load_tokenizer()
    data = load_split(split)

    n_sent = n_char = 0
    max_tokens = 0
    too_long = []
    ambiguous = []          # (对话ID, 句子ID, 原文, 词片信息)
    mismatch = []
    consistent = 0
    n_amb_sent = 0
    n_amb_chars = 0

    for did, d in data.items():
        for s in d["dialogue"]:
            text = s["sentence"]
            tags = s["BIO_label"].split(" ") if has_labels else None
            where = f"对话{did}/句{s['sentence_id']}"
            try:
                enc = encode_sentence(tokenizer, text, max_len=MAX_LEN, tags=tags, where=where)
            except AlignmentError as e:
                too_long.append(str(e))
                continue

            n_sent += 1
            n_char += len(text)
            max_tokens = max(max_tokens, enc.n_tokens)

            if enc.ambiguous:
                n_amb_sent += 1
                for a in enc.ambiguous:
                    n_amb_chars += a["end"] - a["start"]
                    ambiguous.append((did, s["sentence_id"], text, a))

            if not has_labels:
                continue

            restored = restore_char_labels(enc.token_labels, enc.offsets, len(text))
            if any(t is None for t in restored):
                # 只有歧义位置会是 None，上面已经记过了
                continue
            if restored != tags:
                diff = [i for i, (a, b) in enumerate(zip(tags, restored)) if a != b]
                mismatch.append((did, s["sentence_id"], text, diff[:5],
                                 [tags[i] for i in diff[:5]], [restored[i] for i in diff[:5]]))
            else:
                consistent += 1

    print(f"=== {split} ===")
    print(f"句子 {n_sent}  字符 {n_char}  最长输入 {max_tokens} 词片（上限 {MAX_LEN}）")
    print(f"超长报错 {len(too_long)} 句")
    for t in too_long[:SHOW]:
        print("   ", t)
    if has_labels:
        print(f"歧义词片 {len(ambiguous)} 个，涉及 {n_amb_chars} 个字符、{n_amb_sent} 句")
        print(f"其余 {consistent} 句逐字还原一致")
        print(f"非歧义但还原不一致 {len(mismatch)} 句")
        for m in mismatch[:SHOW]:
            print("   例:", m)
        for did, sid, text, a in ambiguous[:SHOW]:
            print(f"   歧义: 对话{did}/句{sid} {text!r} 词片 {a['text']!r} "
                  f"覆盖 {a['start']}-{a['end']}，字符标签 {a['tags']}")

    report = {
        "split": split,
        "sentences": n_sent,
        "chars": n_char,
        "max_tokens": max_tokens,
        "max_len": MAX_LEN,
        "too_long_errors": len(too_long),
        "ambiguous_tokens": len(ambiguous),
        "ambiguous_chars": n_amb_chars,
        "ambiguous_sentences": n_amb_sent,
        "consistent_sentences": consistent,
        "non_ambiguous_mismatches": len(mismatch),
        "ambiguous_detail": [
            {"dialogue_id": d, "sentence_id": s, "sentence": t, **a}
            for d, s, t, a in ambiguous
        ],
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / f"alignment_report_{split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告写入", out)

    assert not too_long, "有句子超长（应为 0）"
    assert not mismatch, "非歧义句子还原不一致"
    print("全部检查通过")


if __name__ == "__main__":
    main()
