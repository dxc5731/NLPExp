# -*- coding: utf-8 -*-
"""实体级 P/R/F1（等价官方 task1/eval_task1.py 口径）。

口径：把逐字 BIO 还原成实体 span (起, 止, 类型)，预测与标准必须
起止位置和类型全等才算命中。micro 平均。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ENTITY_TYPES = ["Symptom", "Drug", "Drug_Category", "Medical_Examination", "Operation"]


def spans(tags):
    out, i, n = [], 0, len(tags)
    while i < n:
        if tags[i].startswith("B-"):
            etype = tags[i][2:]
            j = i + 1
            while j < n and tags[j] == "I-" + etype:
                j += 1
            out.append((i, j, etype))
            i = j
        else:
            i += 1
    return out


def main():
    pred = json.load(open(os.path.join(OUT, "pred_rule.json"), encoding="utf-8"))
    gold = json.load(open(os.path.join(OUT, "val_gold.json"), encoding="utf-8"))

    tp = fp = fn = 0
    per = {t: [0, 0, 0] for t in ENTITY_TYPES}  # tp, fp, fn
    for did in gold:
        for sid in gold[did]:
            g = set(spans(gold[did][sid].split(" ")))
            p = set(spans(pred[did][sid].split(" ")))
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
            for s in g & p:
                per[s[2]][0] += 1
            for s in p - g:
                per[s[2]][1] += 1 if s[2] in per else 0
            for s in g - p:
                per[s[2]][2] += 1 if s[2] in per else 0

    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    print(f"实体数: 标准={tp + fn} 预测={tp + fp} 命中={tp}")
    print(f"实体级 P={P:.4f}  R={R:.4f}  F1={F1:.4f}")
    print(f"\n{'类型':<24}{'P':>8}{'R':>8}{'F1':>8}")
    for t in ENTITY_TYPES:
        a, b, c = per[t]
        p = a / (a + b) if a + b else 0.0
        r = a / (a + c) if a + c else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        print(f"{t:<24}{p:>8.4f}{r:>8.4f}{f:>8.4f}")


if __name__ == "__main__":
    main()
