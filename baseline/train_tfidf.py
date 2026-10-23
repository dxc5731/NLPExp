# -*- coding: utf-8 -*-
"""TF-IDF 小模型：句级 5 类实体有无（纯 numpy Rocchio 中心分类）。

只用 split_v1.json 的 train 训练、val 评估；dev 不碰。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(HERE), "bert_api", "code")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, HERE)

from tfidf_baseline import predict, train_ovr  # noqa: E402

ARTIFACTS = os.path.join(CODE_DIR, "artifacts")
ENTITY_TYPES = ["Symptom", "Drug", "Drug_Category", "Medical_Examination", "Operation"]


def sent_labels(tags):
    present = set()
    for t in tags:
        if t.startswith("B-") or t.startswith("I-"):
            present.add(t[2:])
    return [1 if e in present else 0 for e in ENTITY_TYPES]


def load_xy(split_name, dialogue_ids):
    from src.data_io import load_split
    from src.dataset import collect_examples

    data = load_split(split_name)
    ids = [i for i in dialogue_ids if i in data]
    examples = collect_examples(data, ids)
    return [ex["text"] for ex in examples], [sent_labels(ex["tags"]) for ex in examples]


def main():
    with open(os.path.join(ARTIFACTS, "split_v1.json"), encoding="utf-8") as f:
        split = json.load(f)

    print("训练集加载中...")
    Xtr, Ytr = load_xy("train", split["train"])
    print(f"train 句子数={len(Xtr)}")
    print("val 加载中...")
    Xva, Yva = load_xy("train", split["val"])
    print(f"val 句子数={len(Xva)}\n")

    print("训练 TF-IDF Rocchio 中心分类器（char 2-4gram, 20000 维，阈值 0.05）...")
    vocab, idf, centers = train_ovr(Xtr, Ytr)
    P = predict(vocab, idf, centers, Xva, thresh=0.05)

    n = len(Yva)
    exact = sum(1 for y, p in zip(Yva, P) if list(y) == list(p)) / n
    print(f"\n句级精确匹配（5 类全对）= {exact:.4f}")
    print(f"\n{'类型':<24}{'P':>8}{'R':>8}{'F1':>8}{'支持':>8}")
    for j, t in enumerate(ENTITY_TYPES):
        tp = sum(1 for y, p in zip(Yva, P) if y[j] == 1 and p[j] == 1)
        fp = sum(1 for y, p in zip(Yva, P) if y[j] == 0 and p[j] == 1)
        fn = sum(1 for y, p in zip(Yva, P) if y[j] == 1 and p[j] == 0)
        sup = sum(y[j] for y in Yva)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        print(f"{t:<24}{pr:>8.4f}{rc:>8.4f}{f1:>8.4f}{sup:>8d}")

    out = os.path.join(HERE, "out", "tfidf_val.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"exact_match": exact}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n结果已存：{out}")


if __name__ == "__main__":
    main()
