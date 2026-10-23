# -*- coding: utf-8 -*-
"""min_count 参数对比：词典大小对规则基线抽取效果的影响。

要回答的问题（课程第 2 讲"分词/子词与词表选择对抽取的影响"）：
    词典里的每个词条都来自 train 的标注。min_count 决定"一个词至少在 train 里
    出现几次才收进词典"。
    - min_count 越大 -> 词典越小 -> 短词/噪声词被滤掉 -> 精确率可能上升，
      但没见过的口语变体也进不来 -> 召回率可能下降；
    - min_count 越小 -> 词典越大 -> 召回率可能上升，但一次性词也进来 -> 误标变多。

    只用 train 统计频次，val 评估，全程不碰 dev。

产出：
    out/dict_stats_min<N>.json  每个 min_count 的词典规模
    out/sweep_mincount.json    每个 min_count 的完整指标
    out/sweep_mincount_log.txt 表格（由调用方重定向）

用法：
    python baseline/sweep_mincount.py                  # 默认网格
    python baseline/sweep_mincount.py 1 2 5 10         # 自定义
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(HERE), "bert_api", "code")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, HERE)

from eval_pred import char_level, entity_level  # noqa: E402
from fast_match import all_types, build_index, collect_candidates, select_bio  # noqa: E402
from run_rule import load_examples, load_lexicon, load_split_ids  # noqa: E402

OUT = os.path.join(HERE, "out")
DEFAULT_GRID = [1, 2, 3, 5, 10, 20]


def build_dicts_with_min_count(examples, min_count):
    """复用 extract_dict.py 的取词逻辑，按 min_count 过滤。"""
    from extract_dict import build_dicts
    kept, _ = build_dicts(examples, min_count=min_count)
    return kept


def main(grid=None):
    grid = grid or DEFAULT_GRID
    split = load_split_ids()
    examples = load_examples(split["val"])
    gold_all = [ex["tags"] for ex in examples]

    # train 侧的句子只用来统计词典频次
    from src.data_io import load_split
    from src.dataset import collect_examples
    train_data = load_split("train")
    train_examples = collect_examples(train_data, [i for i in split["train"] if i in train_data])
    print(f"train 句子 {len(train_examples)}（统计词典） / val 句子 {len(examples)}（评估）\n")

    print(f"{'min_cnt':>8}{'词条数':>8}{'丢弃':>8}{'实体P':>9}{'实体R':>9}{'实体F1':>9}"
          f"{'逐字F1':>9}{'ΔF1':>9}")
    print("-" * 70)

    results = []
    best = None
    base_total = None
    for mc in grid:
        kept = build_dicts_with_min_count(train_examples, mc)
        # 跨类冲突处理：与 run_rule.load_lexicon 保持一致 —— 同一词在多个类出现时，
        # 先出现的类保留，其余丢弃；冲突词只计一次（用 set，不能按次数计，
        # 否则一个同时属于 3 类的词会被算成 2 次冲突，和 run_rule 的 97 对不上）。
        owner = {}
        conflict_words = set()
        for etype, items in kept.items():
            for word, cnt in items:
                if word in owner and owner[word] != etype:
                    conflict_words.add(word)
                    continue
                owner[word] = etype
        total = len(owner)
        conflicts = len(conflict_words)
        if base_total is None:
            base_total = total

        index = build_index(owner)
        allow = all_types(index)
        pred_all = [select_bio(ex["text"], collect_candidates(ex["text"], index), allow)
                    for ex in examples]
        pairs = [(ex["dialogue_id"], ex["sentence_id"], g, p)
                 for ex, g, p in zip(examples, gold_all, pred_all)]
        en = entity_level(pairs)
        ch = char_level(pairs)
        results.append({"min_count": mc, "n_words": total, "conflicts": conflicts,
                        "entity": en, "char": ch})

        delta = en["F1"] - (results[0]["entity"]["F1"] if len(results) > 1 else en["F1"])
        print(f"{mc:>8}{total:>8}{conflicts:>8}{en['P']:>9.4f}{en['R']:>9.4f}{en['F1']:>9.4f}"
              f"{ch['F1']:>9.4f}{delta:>+9.4f}")
        if best is None or en["F1"] > best["entity"]["F1"]:
            best = results[-1]

    print("-" * 70)
    print(f"最佳 min_count={best['min_count']}  实体 F1={best['entity']['F1']:.4f}"
          f"（词典 {best['n_words']} 词）")
    r0, r1 = results[0], results[-1]
    print(f"min_count {r0['min_count']} → {r1['min_count']}：词典 {r0['n_words']} → {r1['n_words']} 词，"
          f"精确率 {r0['entity']['P']:.4f} → {r1['entity']['P']:.4f}，"
          f"召回率 {r0['entity']['R']:.4f} → {r1['entity']['R']:.4f}，"
          f"F1 {r0['entity']['F1']:.4f} → {r1['entity']['F1']:.4f}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sweep_mincount.json"), "w", encoding="utf-8") as f:
        json.dump({"grid": grid, "results": results, "best_min_count": best["min_count"]},
                  f, ensure_ascii=False, indent=2)
    print(f"结果已存：{os.path.join(OUT, 'sweep_mincount.json')}")
    print(f"SUMMARY sweep_mincount best_min_count={best['min_count']}"
          f" entity_F1={best['entity']['F1']:.4f} n_words={best['n_words']}")


if __name__ == "__main__":
    g = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(grid=g)
