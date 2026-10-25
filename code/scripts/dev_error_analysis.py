# -*- coding: utf-8 -*-
"""在 dev 上做错误分类，产出"事实和样例"，不做归因。

四类错误（按金标实体对预测实体匹配）：
  类型错   同一位置、同一跨度，类型不同
  边界错   跨度有重叠但不完全相同（多切/少切一个字）
  漏识别   金标有、预测完全没有对应
  误识别   预测有、金标没有（单独统计）

用法（在 code/ 目录下）：
    python -X utf8 -m scripts.dev_error_analysis --predictions <predictions_dev.json> --name BERT-Softmax
"""

import argparse
import json
from collections import Counter

from src.data_io import load_split
from src.labels import ENTITY_TYPES


def spans(tags):
    """BIO 标签串 -> [(start, end, type)]，end 不含。"""
    out = []
    i = 0
    while i < len(tags):
        t = tags[i]
        if t.startswith("B-"):
            ent = t[2:]
            j = i + 1
            while j < len(tags) and tags[j] == "I-" + ent:
                j += 1
            out.append((i, j, ent))
            i = j
        else:
            i += 1
    return out


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pred = json.loads(open(args.predictions, encoding="utf-8").read())
    data = load_split("dev")

    counts = Counter()
    examples = {k: [] for k in ("类型错", "边界错", "漏识别", "误识别")}
    per_type = Counter()
    n_gold = 0

    for did, d in data.items():
        for s in d["dialogue"]:
            sid = s["sentence_id"]
            text = s["sentence"]
            gold = spans(s["BIO_label"].split(" "))
            preds = spans(pred[did][sid].split(" "))
            n_gold += len(gold)
            used = set()

            for g in gold:
                # 1. 完全命中
                if g in preds:
                    used.add(preds.index(g))
                    counts["命中"] += 1
                    continue
                # 2. 同跨度不同类型
                same = [k for k, p in enumerate(preds) if p[0] == g[0] and p[1] == g[1]]
                if same:
                    k = same[0]
                    used.add(k)
                    counts["类型错"] += 1
                    per_type[g[2]] += 1
                    if len(examples["类型错"]) < args.examples:
                        examples["类型错"].append({
                            "对话": did, "句子": sid, "原文": text,
                            "金标": f"{text[g[0]:g[1]]}（{g[2]}）",
                            "预测": f"{text[preds[k][0]:preds[k][1]]}（{preds[k][2]}）",
                        })
                    continue
                # 3. 有重叠但跨度不同
                near = [k for k, p in enumerate(preds) if overlap(g, p) > 0]
                if near:
                    k = near[0]
                    used.add(k)
                    counts["边界错"] += 1
                    per_type[g[2]] += 1
                    if len(examples["边界错"]) < args.examples:
                        examples["边界错"].append({
                            "对话": did, "句子": sid, "原文": text,
                            "金标": f"{text[g[0]:g[1]]}（{g[2]}，{g[0]}-{g[1]}）",
                            "预测": f"{text[preds[k][0]:preds[k][1]]}（{preds[k][2]}，{preds[k][0]}-{preds[k][1]}）",
                        })
                    continue
                # 4. 完全没找到
                counts["漏识别"] += 1
                per_type[g[2]] += 1
                if len(examples["漏识别"]) < args.examples:
                    examples["漏识别"].append({
                        "对话": did, "句子": sid, "原文": text,
                        "金标": f"{text[g[0]:g[1]]}（{g[2]}）",
                        "预测": "无",
                    })

            for k, p in enumerate(preds):
                if k not in used:
                    counts["误识别"] += 1
                    if len(examples["误识别"]) < args.examples:
                        examples["误识别"].append({
                            "对话": did, "句子": sid, "原文": text,
                            "金标": "无",
                            "预测": f"{text[p[0]:p[1]]}（{p[2]}）",
                        })

    total_err = counts["类型错"] + counts["边界错"] + counts["漏识别"]
    print(f"=== {args.name} · dev ===")
    print(f"金标实体 {n_gold}，命中 {counts['命中']}，错 {total_err}，误识别 {counts['误识别']}")
    print(f"  类型错 {counts['类型错']}  边界错 {counts['边界错']}  漏识别 {counts['漏识别']}")
    print(f"  各类错误占比（错误实体的金标类型）：{dict(per_type.most_common())}")
    for k, v in examples.items():
        print(f"\n--- {k} 样例 ---")
        for e in v:
            print(f"  [{e['对话']}/{e['句子']}] {e['原文']}")
            print(f"    金标: {e['金标']}   预测: {e['预测']}")

    if args.out:
        payload = {"method": args.name, "counts": dict(counts), "error_by_type": dict(per_type),
                   "examples": examples, "gold_total": n_gold}
        open(args.out, "w", encoding="utf-8").write(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
