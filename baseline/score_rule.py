# -*- coding: utf-8 -*-
"""用已生成的 pred_rule.json / val_gold.json 算逐字 P/R/F1（不重新匹配，秒出）。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rule_baseline import token_prf

OUT = os.path.join(HERE, "out")
pred = json.load(open(os.path.join(OUT, "pred_rule.json"), encoding="utf-8"))
gold = json.load(open(os.path.join(OUT, "val_gold.json"), encoding="utf-8"))

assert set(pred) == set(gold), "预测与标准对话集合不一致"
gold_all, pred_all, n_sent = [], [], 0
for did in gold:
    assert set(pred[did]) == set(gold[did]), did
    for sid in gold[did]:
        g = gold[did][sid].split(" ")
        p = pred[did][sid].split(" ")
        assert len(g) == len(p), (did, sid)
        gold_all.append(g)
        pred_all.append(p)
        n_sent += 1

m = token_prf(gold_all, pred_all)
print(f"对话数={len(gold)} 句子数={n_sent}")
print(f"逐字准确率 = {m['token_acc']:.4f}")
print(f"逐字 P={m['P']:.4f}  R={m['R']:.4f}  F1={m['F1']:.4f}")
