# -*- coding: utf-8 -*-
"""词典 min_count × gated 阈值 的二维参数对比。

为什么要二维：两个超参都在改同一件事——"规则允许标哪些字"。
  - min_count 决定词表里有多少词（进而决定短噪声词是否进得来）；
  - gated 的 θ 决定 TF-IDF 守门员放行哪些类。
单独扫任一个都会漏掉它们的交互：词表更干净之后，守门员需要放行的类可能变了。

一律按 **实体级 F1** 选参（见 RESULTS.md §3.3：句级目标选出来的阈值在实体级上会掉 0.14）。

产出：
    out/pred_gated_mc<N>_t<θ>.json  每个组合一份预测（不覆盖）
    out/sweep_grid.json              完整结果
    out/sweep_grid_log.txt           表格（由调用方重定向）

用法：
    python baseline/sweep_grid.py
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
from tfidf_baseline import train_ovr, tfidf_row  # noqa: E402
from train_tfidf import ENTITY_TYPES, load_xy  # noqa: E402

OUT = os.path.join(HERE, "out")
MC_GRID = [1, 2, 3, 5, 10, 20]
THETA_GRID = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]


def main(mc_grid=None, theta_grid=None):
    mc_grid = mc_grid or MC_GRID
    theta_grid = theta_grid or THETA_GRID

    split = load_split_ids()
    examples = load_examples(split["val"])
    gold_all = [ex["tags"] for ex in examples]

    print("训练 TF-IDF 守门员（只此一次）...")
    Xtr, Ytr = load_xy("train", split["train"])
    vocab, idf, centers = train_ovr(Xtr, Ytr)

    scores = []
    for ex in examples:
        v = tfidf_row(ex["text"], vocab, idf)
        scores.append([sum(val * c[k] for k, val in v.items()) for c in centers])

    print(f"\n实体级 F1 矩阵（行=min_count，列=θ）")
    header = "".join(f"{t:>10.3f}" for t in theta_grid)
    print(f"{'min_cnt':>8}{header}")
    print("-" * (8 + 10 * len(theta_grid)))

    os.makedirs(OUT, exist_ok=True)
    rows = []
    best = None
    for mc in mc_grid:
        lexicon, _ = load_lexicon(min_count=mc)
        index = build_index(lexicon)
        allow = all_types(index)
        hits = [collect_candidates(ex["text"], index) for ex in examples]
        line = f"{mc:>8}"
        for th in theta_grid:
            pred_all = []
            for ex, h, sc in zip(examples, hits, scores):
                present = {ENTITY_TYPES[j] for j, s in enumerate(sc) if s >= th}
                pred_all.append(select_bio(ex["text"], h, present))
            pairs = [(ex["dialogue_id"], ex["sentence_id"], g, p)
                     for ex, g, p in zip(examples, gold_all, pred_all)]
            en = entity_level(pairs)
            ch = char_level(pairs)
            rows.append({"min_count": mc, "theta": th, "n_words": len(lexicon),
                         "entity": en, "char": ch})
            line += f"{en['F1']:>10.4f}"
            if best is None or en["F1"] > best["entity"]["F1"]:
                best = rows[-1]
                best_pred = pred_all
        print(line)

    print("-" * (8 + 10 * len(theta_grid)))
    print(f"最佳：min_count={best['min_count']}  θ={best['theta']:.3f}  "
          f"词典 {best['n_words']} 词  实体 F1={best['entity']['F1']:.4f} "
          f"(P={best['entity']['P']:.4f} R={best['entity']['R']:.4f})")
    print(f"      逐字 F1={best['char']['F1']:.4f} acc={best['char']['acc']:.4f} "
          f"(tp={best['entity']['tp']} fp={best['entity']['fp']} fn={best['entity']['fn']})")

    # 最佳组合的预测单独存一份，和其他组合互不覆盖
    pj = {}
    for ex, tags in zip(examples, best_pred):
        pj.setdefault(ex["dialogue_id"], {})[ex["sentence_id"]] = " ".join(tags)
    name = f"pred_gated_mc{best['min_count']}_t{best['theta']:.3f}.json"
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(pj, f, ensure_ascii=False)
    print(f"最佳组合预测已存：out/{name}")

    with open(os.path.join(OUT, "sweep_grid.json"), "w", encoding="utf-8") as f:
        json.dump({"mc_grid": mc_grid, "theta_grid": theta_grid, "results": rows,
                   "best": {k: v for k, v in best.items() if k != "char"} |
                             {"char": best["char"]}}, f, ensure_ascii=False, indent=2)
    print(f"结果已存：{os.path.join(OUT, 'sweep_grid.json')}")
    print(f"SUMMARY sweep_grid best_min_count={best['min_count']} best_theta={best['theta']:.3f}"
          f" entity_F1={best['entity']['F1']:.4f} n_words={best['n_words']}")


if __name__ == "__main__":
    main()
