# -*- coding: utf-8 -*-
"""从 train 划分提取 5 类医疗实体词典（只用 split_v1.json 的 train 名单）。

用法（在 bert_api/code 目录下或已 pip install 本包后）：
    python -m baseline.extract_dict
或直接：
    python baseline/extract_dict.py

产出：
    baseline/dict/dict_symptom.txt ... 5 个词表，每行 "词 频次"
    baseline/dict/dict_stats.json  词表统计，写进报告用
"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(HERE), "bert_api", "code")
sys.path.insert(0, CODE_DIR)

ARTIFACTS = os.path.join(CODE_DIR, "artifacts")
OUT_DIR = os.path.join(HERE, "dict")

# 11 类标签里的 5 个实体类型（与 labels.py 一致，不要另起一份）
ENTITY_TYPES = ["Symptom", "Drug", "Drug_Category", "Medical_Examination", "Operation"]
TYPE2FILE = {
    "Symptom": "dict_symptom.txt",
    "Drug": "dict_drug.txt",
    "Drug_Category": "dict_drug_category.txt",
    "Medical_Examination": "dict_exam.txt",
    "Operation": "dict_operation.txt",
}


def load_train_examples():
    """读官方 train，按名单过滤成句子清单；每句校验标签数=字数。"""
    from src.data_io import load_split
    from src.dataset import collect_examples

    with open(os.path.join(ARTIFACTS, "split_v1.json"), encoding="utf-8") as f:
        split = json.load(f)
    train_ids = split["train"]

    data = load_split("train")
    missing = [i for i in train_ids if i not in data]
    if missing:
        print(f"[警告] 名单里有 {len(missing)} 个 ID 不在 train.json 中，跳过")
        train_ids = [i for i in train_ids if i in data]

    examples = collect_examples(data, train_ids)
    return train_ids, examples, split


def extract_entities_from_tags(text, tags):
    """把逐字 BIO 拼成实体词。返回 [(词, 类型)]。

    连续 B-X 后跟任意个 I-X 属于同一实体；只有 B-X 的按单字/多字实体算一个词。
    非法格式（如 I 开头）跳过，不猜。
    """
    entities = []
    i, n = 0, len(tags)
    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < n and tags[j] == "I-" + etype:
                j += 1
            word = text[i:j]
            if etype in ENTITY_TYPES and word.strip():
                entities.append((word, etype))
            i = j
        else:
            i += 1
    return entities


def build_dicts(examples, min_count=1):
    """扫全部句子，按类型计词频。"""
    counters = {t: Counter() for t in ENTITY_TYPES}
    sent_with_entity = 0
    for ex in examples:
        ents = extract_entities_from_tags(ex["text"], ex["tags"])
        if ents:
            sent_with_entity += 1
        for word, etype in ents:
            counters[etype][word] += 1

    kept = {}
    for t in ENTITY_TYPES:
        kept[t] = [(w, c) for w, c in counters[t].items() if c >= min_count]
        kept[t].sort(key=lambda x: (-x[1], x[0]))
    return kept, sent_with_entity


def save_dicts(kept):
    os.makedirs(OUT_DIR, exist_ok=True)
    for t, items in kept.items():
        path = os.path.join(OUT_DIR, TYPE2FILE[t])
        with open(path, "w", encoding="utf-8") as f:
            for word, cnt in items:
                f.write(f"{word} {cnt}\n")
    return {t: os.path.join(OUT_DIR, TYPE2FILE[t]) for t in ENTITY_TYPES}


def main(min_count=1):
    print("=" * 60)
    print("IMCS21 五类实体词典提取（只用 split_v1.json 的 train 名单）")
    print("=" * 60)

    train_ids, examples, split = load_train_examples()
    print(f"seed={split['seed']} val_ratio={split['val_ratio']}")
    print(f"train 对话数={len(train_ids)}  句子数={len(examples)}")

    kept, sent_with_entity = build_dicts(examples, min_count=min_count)
    paths = save_dicts(kept)

    stats = {}
    print(f"\n含实体的句子数={sent_with_entity} ({sent_with_entity / max(1, len(examples)):.1%})")
    print(f"\n{'类型':<24}{'词条数':>8}   高频例词")
    print("-" * 60)
    for t in ENTITY_TYPES:
        items = kept[t]
        total = sum(c for _, c in items)
        top = [w for w, _ in items[:8]]
        stats[t] = {
            "n_words": len(items),
            "total_mentions": total,
            "top10": items[:10],
        }
        print(f"{t:<24}{len(items):>8}   {'、'.join(top)}")
        print(f"{'  文件':<24}{'':>8}   {paths[t]}")

    with open(os.path.join(OUT_DIR, "dict_stats.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"seed": split["seed"], "n_sentences": len(examples),
             "n_train_dialogues": len(train_ids), "stats": stats},
            f, ensure_ascii=False, indent=2)
    print(f"\n统计已存：{os.path.join(OUT_DIR, 'dict_stats.json')}")
    print("下一步：跑规则基线（在 val 上出实体级 F1 + 逐字 F1）")


if __name__ == "__main__":
    mc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(min_count=mc)