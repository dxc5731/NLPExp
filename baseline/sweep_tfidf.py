# -*- coding: utf-8 -*-
"""阈值参数对比：TF-IDF 二分类阈值对句级精确匹配的影响（任务书要求的参数对比组）。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(HERE), "bert_api", "code")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, HERE)

from tfidf_baseline import train_ovr, tfidf_row  # noqa: E402
from train_tfidf import ENTITY_TYPES, load_xy  # noqa: E402

ARTIFACTS = os.path.join(CODE_DIR, "artifacts")


def main():
    with open(os.path.join(ARTIFACTS, "split_v1.json"), encoding="utf-8") as f:
        split = json.load(f)
    Xtr, Ytr = load_xy("train", split["train"])
    Xva, Yva = load_xy("train", split["val"])
    vocab, idf, centers = train_ovr(Xtr, Ytr)

    vecs = [tfidf_row(x, vocab, idf) for x in Xva]
    scores = []
    for v in vecs:
        scores.append([sum(val * c[k] for k, val in v.items()) for c in centers])

    print(f"{'阈值':>6}{'句级精确':>10}" + "".join(f"{t[:8]:>10}" for t in ENTITY_TYPES))
    best = None
    for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        P = [[1 if s[j] >= th else 0 for j in range(len(ENTITY_TYPES))] for s in scores]
        exact = sum(1 for y, p in zip(Yva, P) if list(y) == list(p)) / len(Yva)
        f1s = []
        for j in range(len(ENTITY_TYPES)):
            tp = sum(1 for y, p in zip(Yva, P) if y[j] == 1 and p[j] == 1)
            fp = sum(1 for y, p in zip(Yva, P) if y[j] == 0 and p[j] == 1)
            fn = sum(1 for y, p in zip(Yva, P) if y[j] == 1 and p[j] == 0)
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        line = f"{th:>6.2f}{exact:>10.4f}" + "".join(f"{f:>10.4f}" for f in f1s)
        print(line)
        if best is None or exact > best[1]:
            best = (th, exact)
    print(f"\n最佳阈值={best[0]} 句级精确={best[1]:.4f}")


if __name__ == "__main__":
    main()
