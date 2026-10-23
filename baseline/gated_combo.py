# -*- coding: utf-8 -*-
"""Gated 组合基线：TF-IDF 判类当守门员 → 规则词表定位描边。

对比：纯规则 / 纯 TF-IDF / gated 组合，同一份 val，实体级 P/R/F1。
只用 split_v1.json 的 train 训练 TF-IDF、val 评估；dev 不碰。

原理：规则精确低是因为短词到处误标；TF-IDF 召回高但不知位置。
gated = TF-IDF 决定"这句有哪些类" → 规则只在这些类里最长匹配，
压掉规则对未判出类的误标，同时保留规则给的字符边界。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(HERE), "bert_api", "code")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, HERE)

from run_rule import load_lexicon, load_examples, load_split_ids  # noqa: E402
from tfidf_baseline import train_ovr, tfidf_row  # noqa: E402
from train_tfidf import ENTITY_TYPES, load_xy  # noqa: E402
from eval_pred import char_level, entity_level  # noqa: E402  统一口径
from fast_match import all_types, build_index, collect_candidates, select_bio  # noqa: E402

OUT = os.path.join(HERE, "out")


def main(thresh=0.10, tag=None, min_count=1):
    lexicon, _ = load_lexicon(min_count=min_count)
    split = load_split_ids()
    examples = load_examples(split["val"])
    gold_all = [ex["tags"] for ex in examples]
    print(f"词典 min_count={min_count}：{len(lexicon)} 词")

    # 索引建一次、候选扫一次；select_bio 的语义等价于"在子词表上跑 longest_match_bio"
    index = build_index(lexicon)
    allow = all_types(index)
    hits = [collect_candidates(ex["text"], index) for ex in examples]

    # 纯规则（不限类）
    rule_all = [select_bio(ex["text"], h, allow) for ex, h in zip(examples, hits)]

    # 训练 TF-IDF
    print("训练 TF-IDF 守门员...")
    Xtr, Ytr = load_xy("train", split["train"])
    vocab, idf, centers = train_ovr(Xtr, Ytr)

    # gated
    gated_all = []
    for ex, h in zip(examples, hits):
        v = tfidf_row(ex["text"], vocab, idf)
        sc = [sum(val * c[k] for k, val in v.items()) for c in centers]
        present = {ENTITY_TYPES[j] for j, s in enumerate(sc) if s >= thresh}
        gated_all.append(select_bio(ex["text"], h, present))

    # TF-IDF 单独（句级有无 → 没法给 BIO，只能把整句归为命中的类别，这里不评实体级）
    rule_pairs = [(ex["dialogue_id"], ex["sentence_id"], g, p)
                  for ex, g, p in zip(examples, gold_all, rule_all)]
    gated_pairs = [(ex["dialogue_id"], ex["sentence_id"], g, p)
                   for ex, g, p in zip(examples, gold_all, gated_all)]

    print(f"\n阈值={thresh}（统一口径，定义见 baseline/eval_pred.py 文件头）")
    for name, prs in (("纯规则", rule_pairs), ("gated 组合", gated_pairs)):
        ch = char_level(prs)
        en = entity_level(prs)
        print(f"\n--- {name} ---")
        print(f"  逐字   P={ch['P']:.4f} R={ch['R']:.4f} F1={ch['F1']:.4f} acc={ch['acc']:.4f}")
        print(f"  实体   P={en['P']:.4f} R={en['R']:.4f} F1={en['F1']:.4f}"
              f"  (tp={en['tp']} fp={en['fp']} fn={en['fn']})")

    os.makedirs(OUT, exist_ok=True)
    pred_json = {}
    for ex, tags in zip(examples, gated_all):
        pred_json.setdefault(ex["dialogue_id"], {})[ex["sentence_id"]] = " ".join(tags)
    pred_name = "pred_gated.json" if not tag else f"pred_gated_{tag}.json"
    pred_path = os.path.join(OUT, pred_name)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_json, f, ensure_ascii=False)
    print(f"\ngated 预测已存：{pred_path}")


if __name__ == "__main__":
    th = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
    tg = sys.argv[2] if len(sys.argv) > 2 else None
    mc = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    main(thresh=th, tag=tg, min_count=mc)
