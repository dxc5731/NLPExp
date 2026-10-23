# -*- coding: utf-8 -*-
"""用已有预测文件算实体级 P/R/F1（不重训）。用法：python score_pred.py <pred.json>"""
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
            et = tags[i][2:]
            j = i + 1
            while j < n and tags[j] == "I-" + et:
                j += 1
            out.append((i, j, et))
            i = j
        else:
            i += 1
    return out


def main(pred_name):
    pred = json.load(open(os.path.join(OUT, pred_name), encoding="utf-8"))
    gold = json.load(open(os.path.join(OUT, "val_gold.json"), encoding="utf-8"))
    tp = fp = fn = 0
    for did in gold:
        for sid in gold[did]:
            g = set(spans(gold[did][sid].split(" ")))
            p = set(spans(pred[did][sid].split(" ")))
            tp += len(g & p); fp += len(p - g); fn += len(g - p)
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    F1 = 2 * P * R / (P + R) if P + R else 0
    print(f"{pred_name}: P={P:.4f} R={R:.4f} F1={F1:.4f} (tp={tp} fp={fp} fn={fn})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pred_gated.json")
