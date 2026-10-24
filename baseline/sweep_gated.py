# -*- coding: utf-8 -*-
"""gated 阈值扫描：θ 是"本句允许出现哪些类"的门槛，直接决定 gated 的实体级 F1。

为什么要专门扫：
  原来 θ=0.10 是拿"句级精确匹配"挑出来的（见 sweep_tfidf.py），但 gated 的产出和
  评价都是**实体级 F1** —— 优化目标和评价口径不是一个东西，这就是口径没闭环。
  本脚本按实体级 F1 重新扫，谁最优用它。

用 fast_match 加速（已验证与旧实现逐句等价），一轮扫描秒级完成。

用法：
    python baseline/sweep_gated.py                       # 默认网格
    python baseline/sweep_gated.py 0.02 0.04 0.06 0.08   # 自定义网格
产出：
    sweeps/pred_gated_<θ>.json   每个 θ 一份预测（不覆盖，便于核对）
    sweeps/sweep_gated.json      机器可读结果
    sweeps/sweep_gated_log.txt   表格（由调用方重定向）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(HERE), "bert_api", "code")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, HERE)

from eval_pred import char_level, entity_level  # noqa: E402
from fast_match import build_index, collect_candidates, select_bio  # noqa: E402
from run_rule import load_lexicon, load_examples, load_split_ids  # noqa: E402
from tfidf_baseline import train_ovr, tfidf_row  # noqa: E402
from train_tfidf import ENTITY_TYPES, load_xy  # noqa: E402
from src.labels import ENTITY_TYPES as LABEL_TYPES  # noqa: E402

OUT = os.path.join(HERE, "sweeps")
DEFAULT_GRID = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]


def main(grid=None):
    grid = grid or DEFAULT_GRID
    assert list(ENTITY_TYPES) == list(LABEL_TYPES), "两类标签顺序必须一致"

    lexicon, _ = load_lexicon()
    index = build_index(lexicon)
    split = load_split_ids()
    examples = load_examples(split["val"])
    print(f"词表 {len(lexicon)} 词；val {len(examples)} 句")

    # 候选只扫一次，后面每个 θ 都复用
    hits = [collect_candidates(ex["text"], index) for ex in examples]
    gold_all = [ex["tags"] for ex in examples]

    print("训练 TF-IDF 守门员（只此一次）...")
    Xtr, Ytr = load_xy("train", split["train"])
    vocab, idf, centers = train_ovr(Xtr, Ytr)

    # 每句每类的余弦分，也只算一次
    scores = []
    for ex in examples:
        v = tfidf_row(ex["text"], vocab, idf)
        scores.append([sum(val * c[k] for k, val in v.items()) for c in centers])

    # 参照：纯规则（不限类）
    alltypes = set(ENTITY_TYPES)
    rule_all = [select_bio(ex["text"], h, alltypes) for ex, h in zip(examples, hits)]
    rule_pairs = [(ex["dialogue_id"], ex["sentence_id"], g, p)
                  for ex, g, p in zip(examples, gold_all, rule_all)]
    rule_en = entity_level(rule_pairs)
    rule_ch = char_level(rule_pairs)
    print(f"\n参照 纯规则：实体 F1={rule_en['F1']:.4f}  逐字 F1={rule_ch['F1']:.4f}\n")

    print(f"{'θ':>6}{'实体P':>9}{'实体R':>9}{'实体F1':>9}{'逐字F1':>9}{'ΔF1':>9}{'预测实体':>10}")
    print("-" * 62)
    os.makedirs(OUT, exist_ok=True)
    results = []
    best = None
    for th in grid:
        pred_all = []
        for ex, h, sc in zip(examples, hits, scores):
            present = {ENTITY_TYPES[j] for j, s in enumerate(sc) if s >= th}
            pred_all.append(select_bio(ex["text"], h, present))
        pairs = [(ex["dialogue_id"], ex["sentence_id"], g, p)
                 for ex, g, p in zip(examples, gold_all, pred_all)]
        en = entity_level(pairs)
        ch = char_level(pairs)
        results.append({"theta": th, "entity": en, "char": ch})

        # 每个 θ 的预测单独存一份，便于核对，互不覆盖
        tag = f"{th:.3f}"
        pj = {}
        for ex, tags in zip(examples, pred_all):
            pj.setdefault(ex["dialogue_id"], {})[ex["sentence_id"]] = " ".join(tags)
        with open(os.path.join(OUT, f"pred_gated_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(pj, f, ensure_ascii=False)

        delta = en["F1"] - rule_en["F1"]
        print(f"{th:>6.3f}{en['P']:>9.4f}{en['R']:>9.4f}{en['F1']:>9.4f}"
              f"{ch['F1']:>9.4f}{delta:>+9.4f}{en['pred_entities']:>10}")
        if best is None or en["F1"] > best["entity"]["F1"]:
            best = results[-1]

    print("-" * 62)
    print(f"最佳 θ={best['theta']:.3f}  实体 F1={best['entity']['F1']:.4f}"
          f"（P={best['entity']['P']:.4f} R={best['entity']['R']:.4f}）")
    print(f"参照 纯规则 实体 F1={rule_en['F1']:.4f}  -> gated {'优于' if best['entity']['F1'] > rule_en['F1'] else '不优于'} 纯规则")
    print(f"各 θ 预测已存：out/pred_gated_<θ>.json")

    with open(os.path.join(OUT, "sweep_gated.json"), "w", encoding="utf-8") as f:
        json.dump({"grid": grid, "rule_entity": rule_en, "rule_char": rule_ch,
                   "results": results, "best_theta": best["theta"]},
                  f, ensure_ascii=False, indent=2)
    print(f"结果已存：{os.path.join(OUT, 'sweep_gated.json')}")
    print(f"SUMMARY sweep_gated best_theta={best['theta']:.3f}"
          f" entity_F1={best['entity']['F1']:.4f} rule_F1={rule_en['F1']:.4f}")


if __name__ == "__main__":
    g = [float(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(grid=g)
