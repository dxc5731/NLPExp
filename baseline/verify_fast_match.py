# -*- coding: utf-8 -*-
"""验证 fast_match 与 rule_baseline.longest_match_bio 完全等价（逐句比对）。

跑法（作业根目录）：
    $env:IMCS_DATA_DIR="...\\IMCS-V2-NER"
    python baseline/verify_fast_match.py

两个测试：
  A. 全词表（5 类不限）对 val 全部 9687 句：fast_match vs out/pred_rule.json
     —— pred_rule.json 是旧实现写出来的，逐句比等于全量等价性验证。
  B. 随机类型子集（固定种子 42）对 500 句：fast_match vs 旧实现当场算
     —— 覆盖 gated 那种"只允许部分类"的用法。

任一测试出现不一致，脚本以非 0 退出，别把加速版当等价版用。
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(HERE), "bert_api", "code")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, HERE)

from fast_match import build_index, all_types, match_one  # noqa: E402
from rule_baseline import longest_match_bio  # noqa: E402
from run_rule import load_lexicon, load_examples, load_split_ids  # noqa: E402
from src.labels import ENTITY_TYPES  # noqa: E402


def main():
    lexicon, _ = load_lexicon()
    index = build_index(lexicon)
    split = load_split_ids()
    examples = load_examples(split["val"])
    print(f"词表 {len(lexicon)} 词；val {len(examples)} 句")

    # ---------------- 测试 A：全词表，和旧实现写出的预测文件逐句比
    with open(os.path.join(HERE, "out", "pred_rule.json"), encoding="utf-8") as f:
        stored = json.load(f)
    bad_a = []
    alltypes = all_types(index)
    for ex in examples:
        fast = match_one(ex["text"], index, alltypes)
        old = stored[ex["dialogue_id"]][ex["sentence_id"]].split()
        if fast != old:
            bad_a.append((ex["dialogue_id"], ex["sentence_id"]))
    print(f"测试A 全词表 vs pred_rule.json：不一致 {len(bad_a)} / {len(examples)} 句")
    if bad_a[:3]:
        print(f"  示例：{bad_a[:3]}")

    # ---------------- 测试 B：随机类型子集，当场和旧实现比
    rng = random.Random(42)
    sample = rng.sample(examples, min(500, len(examples)))
    bad_b = 0
    total = 0
    for ex in sample:
        k = rng.randint(1, len(ENTITY_TYPES))
        sub_types = set(rng.sample(ENTITY_TYPES, k))
        sub_lex = {w: t for w, t in lexicon.items() if t in sub_types}
        old = longest_match_bio(ex["text"], sub_lex)
        fast = match_one(ex["text"], index, sub_types)
        total += 1
        if old != fast:
            bad_b += 1
            if bad_b <= 3:
                print(f"  不一致：{ex['dialogue_id']}/{ex['sentence_id']} 类型={sorted(sub_types)}")
    print(f"测试B 随机类型子集：不一致 {bad_b} / {total} 句")

    ok = (len(bad_a) == 0 and bad_b == 0)
    print(f"\nSUMMARY fast_match 等价性: A不一致={len(bad_a)} B不一致={bad_b} -> {'通过' if ok else '不通过'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
