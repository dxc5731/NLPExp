# -*- coding: utf-8 -*-
r"""官方 dev 推理与统一评分（冻结 mc10 词典，独立输出目录，不覆盖 val 结果）。

为什么单独一个脚本：
    原 `run_rule.py` 把预测写到 `baseline/out/pred_rule*.json`、标准写到
    `val_gold.json`。如果拿它跑 dev，会把 val 的预测/标准覆盖掉，导致
    "val 的 0.7620" 和 "dev 的结果" 混在一起分不清。本脚本：
      - 直接读取**冻结词典**（baseline/dict/ 里已提交的词表，按 min_count 过滤，
        不重新统计、不重新生成），保证结果可复现；
      - 只跑官方 dev split（IMCS-V2_dev.json，833 对话 / 33267 句，带金标准）；
      - 所有产物写入独立目录 `baseline/out_dev/`，绝不碰 out/。

用法（在作业根目录）：
    python baseline/run_dev.py                # 默认 mc10 冻结词典
    python baseline/run_dev.py --min-count 10 # 显式指定
产出：
    baseline/out_dev/pred_rule_min10_dev.json   dev 预测（提交格式）
    baseline/out_dev/dev_gold.json              dev 金标准
    baseline/out_dev/eval_pred_rule_min10_dev.json  统一口径评分
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(HERE), "bert_api", "code")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, HERE)

from fast_match import all_types, build_index, collect_candidates, select_bio  # noqa: E402
from run_rule import load_examples, load_lexicon  # noqa: E402

OUT_DEV = os.path.join(HERE, "out_dev")


def run_dev(min_count=10, show=5):
    os.makedirs(OUT_DEV, exist_ok=True)

    # 冻结词典：直接读 baseline/dict/ 里已提交的词表，按 min_count 过滤
    lexicon, conflicts = load_lexicon(min_count=min_count)
    print(f"冻结词典载入：{len(lexicon)} 词（源 baseline/dict/，min_count={min_count}"
          + (f"，跨类冲突丢弃 {len(conflicts)} 个）" if conflicts else "）"))

    # 官方 dev 划分：用 data_io 的 load_split('dev') + collect_examples
    from src.data_io import load_split
    from src.dataset import collect_examples

    data = load_split("dev")
    examples = collect_examples(data, list(data.keys()))
    print(f"dev 对话数={len(data)}  句子数={len(examples)}\n")

    index = build_index(lexicon)
    allow = all_types(index)
    pred_json = {}
    samples = []
    for ex in examples:
        pred = select_bio(ex["text"], collect_candidates(ex["text"], index), allow)
        assert len(pred) == len(ex["text"]) == len(ex["tags"]), (ex["dialogue_id"], ex["sentence_id"])
        pred_json.setdefault(ex["dialogue_id"], {})[ex["sentence_id"]] = " ".join(pred)
        if len(samples) < show:
            samples.append((ex["text"], ex["tags"], pred))

    base = "pred_rule_min10_dev" if min_count == 10 else f"pred_rule_min{min_count}_dev"
    pred_path = os.path.join(OUT_DEV, base + ".json")
    gold_path = os.path.join(OUT_DEV, "dev_gold.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_json, f, ensure_ascii=False)
    gold_json = {}
    for ex in examples:
        gold_json.setdefault(ex["dialogue_id"], {})[ex["sentence_id"]] = " ".join(ex["tags"])
    with open(gold_path, "w", encoding="utf-8") as f:
        json.dump(gold_json, f, ensure_ascii=False)

    print(f"dev 预测：{pred_path}")
    print(f"dev 标准：{gold_path}")
    print("\n" + "=" * 62)
    print(f"肉眼抽查（前 {len(samples)} 句）")
    print("=" * 62)
    for text, g, p in samples:
        print(f"\n原文: {text}")
        print(f"标准: {' '.join(g)}")
        print(f"预测: {' '.join(p)}")

    print("\n现在用统一口径评分：")
    print(f"    python baseline/eval_pred.py {os.path.relpath(pred_path)}"
          f" --gold {os.path.relpath(gold_path)} --save")
    return pred_path, gold_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="官方 dev 推理（冻结词典，独立 out_dev/）")
    ap.add_argument("--min-count", type=int, default=10, help="词典最低词频，默认 10（冻结档）")
    args = ap.parse_args()
    run_dev(min_count=args.min_count)
