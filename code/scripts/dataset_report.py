# -*- coding: utf-8 -*-
"""数据集体检报告：规模、长度分布、标签分布、BIO 合法性。

这是给报告第⑤节「测试集来源与规模」和第④节「实现细节」用的底稿。
只读，不改任何数据文件。

用法（在 code/ 目录下）：
    python -m scripts.dataset_report
"""

from collections import Counter

from src.data_io import load_split
from src.labels import ENTITY_TYPES, LABELS


def report(split: str, has_label: bool) -> dict:
    data = load_split(split)
    lens, tags, ents = [], Counter(), Counter()
    bad_len = bad_bio = 0
    all_o = 0
    for d in data.values():
        for s in d["dialogue"]:
            text = s["sentence"]
            lens.append(len(text))
            if not has_label:
                continue
            seq = s["BIO_label"].split(" ")
            if len(seq) != len(text):
                bad_len += 1
                continue
            if all(t == "O" for t in seq):
                all_o += 1
            prev = "O"
            for t in seq:
                tags[t] += 1
                if t.startswith("B-"):
                    ents[t[2:]] += 1
                if t.startswith("I-") and prev != "B-" + t[2:] and prev != t:
                    bad_bio += 1
                prev = t
    lens.sort()
    info = {
        "dialogues": len(data),
        "sentences": len(lens),
        "chars": sum(lens),
        "max_len": lens[-1],
        "p99_len": lens[int(len(lens) * 0.99)],
        "all_o_sentences": all_o,
        "bad_len": bad_len,
        "bad_bio": bad_bio,
    }
    print(f"=== {split} ===")
    print(f"对话 {info['dialogues']}  句子 {info['sentences']}  字符 {info['chars']}")
    print(f"最长 {info['max_len']}  p99 {info['p99_len']}")
    if has_label:
        print(f"全 O 句子 {all_o} ({all_o / max(len(lens),1) * 100:.1f}%)")
        print(f"长度不匹配 {bad_len}  非法 BIO {bad_bio}")
        print("标签分布: " + "  ".join(f"{t}={tags[t]}" for t in LABELS if t in tags))
        print("实体分布: " + "  ".join(f"{t}={ents[t]}" for t in ENTITY_TYPES))
    return info


if __name__ == "__main__":
    report("train", True)
    print()
    report("dev", True)
    print()
    report("test", False)
