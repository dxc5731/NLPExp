# -*- coding: utf-8 -*-
"""为错误分析的归因提供可核查的事实：样例回溯、标注规律统计。"""
import json
from collections import Counter

from src.data_io import load_split
from src.labels import ENTITY_TYPES

PRED = r"D:\agent-chat\dsh-deepseek\nlp\runs\lr5e-5_ep3_20261022-151229\predict_dev\predictions_dev.json"


def spans(tags):
    out, i = [], 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            ent, j = tags[i][2:], i + 1
            while j < len(tags) and tags[j] == "I-" + ent:
                j += 1
            out.append((i, j, ent))
            i = j
        else:
            i += 1
    return out


def main():
    train = load_split("train")
    dev = load_split("dev")
    pred = json.loads(open(PRED, encoding="utf-8").read())

    print("=" * 70)
    print("一、三个失败样例回溯（对话ID / 句子ID / 金标 / 预测）")
    targets = [("10190259", "4"), ("10446357", "62"), ("10190259", "12")]
    for did, sid in targets:
        for s in dev[did]["dialogue"]:
            if s["sentence_id"] == sid:
                gold = spans(s["BIO_label"].split(" "))
                p = spans(pred[did][sid].split(" "))
                print(f"\n[{did}/{sid}] {s['sentence']}")
                print("  金标:", [(s["sentence"][a:b], t) for a, b, t in gold])
                print("  预测:", [(s["sentence"][a:b], t) for a, b, t in p])

    print("\n" + "=" * 70)
    print("二、训练集里「度左右 / 度多」这类表达的标注规律")
    total, include_after, stop_at_du = 0, 0, 0
    examples = []
    for did, d in train.items():
        for s in d["dialogue"]:
            t = s["sentence"]
            for kw in ("度左右", "度多", "度上下"):
                idx = t.find(kw)
                while idx != -1:
                    total += 1
                    gold = spans(s["BIO_label"].split(" "))
                    cover = [g for g in gold if g[0] <= idx < g[1] or g[0] <= idx + 1 < g[1]]
                    if cover:
                        if any(g[1] > idx + 1 for g in cover):
                            include_after += 1
                            if len(examples) < 5:
                                examples.append((did, s["sentence_id"], t, "包含修饰词", cover))
                        else:
                            stop_at_du += 1
                            if len(examples) < 10:
                                examples.append((did, s["sentence_id"], t, "止于度", cover))
                    idx = t.find(kw, idx + 1)
    print(f"  出现次数 {total}；实体止于「度」{stop_at_du}；实体包含「左右/多」{include_after}；无实体覆盖 {total - stop_at_du - include_after}")
    for e in examples:
        print(f"    [{e[0]}/{e[1]}] {e[2]}  → {e[3]} {[(e[2][a:b], t) for a, b, t in e[4]]}")

    print("\n" + "=" * 70)
    print("三、训练集里含「贴」「糖浆」的实体，在 Drug 与 Drug_Category 之间怎么分")
    for kw in ("贴", "糖浆", "颗粒", "口服液"):
        c = Counter()
        samples = []
        for did, d in train.items():
            for s in d["dialogue"]:
                t = s["sentence"]
                for a, b, typ in spans(s["BIO_label"].split(" ")):
                    if kw in t[a:b]:
                        c[typ] += 1
                        if len(samples) < 3:
                            samples.append((t[a:b], typ))
        print(f"  「{kw}」: {dict(c)}  例: {samples}")

    print("\n" + "=" * 70)
    print("四、dev 里「发热」的标注情况（含样例 3 那句）")
    as_sym, not_tagged = 0, 0
    for did, d in dev.items():
        for s in d["dialogue"]:
            t = s["sentence"]
            if "发热" not in t:
                continue
            gold = spans(s["BIO_label"].split(" "))
            hit = [g for g in gold if t[g[0]:g[1]] == "发热"]
            if hit:
                as_sym += 1
            else:
                not_tagged += 1
    print(f"  含「发热」的句子中：标为实体的 {as_sym} 句，未标的 {not_tagged} 句")

    for s in dev["10190259"]["dialogue"]:
        if s["sentence_id"] == "12":
            print(f"  样例 3 原句: {s['sentence']}")
            print(f"  整段对话里「发热」出现在:")
    for s in dev["10190259"]["dialogue"]:
        if "发热" in s["sentence"]:
            gold = spans(s["BIO_label"].split(" "))
            print(f"    [{s['sentence_id']}] {s['sentence']}  金标: {[(s['sentence'][a:b], t) for a, b, t in gold]}")


if __name__ == "__main__":
    main()
