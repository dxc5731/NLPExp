# -*- coding: utf-8 -*-
"""统一评测口径：任何方法产出的 pred.json 都用这一把尺子打分。

为什么要有它：原来逐字级在 run_rule.py 里、实体级在 score_pred.py 里、
score_entity.py 又写死了 pred_rule.json —— 三套脚本各管一摊，规则 / gated /
BERT / LLM 没法保证同口径。这个脚本是唯一入口，成员 B 的 pred_bert.json、
pred_llm.json 直接丢进来就能打分，保证主对比表每一行都是同一把尺子量出来的。

输入格式（全项目统一）：
    {对话ID: {句子ID: "O B-Symptom I-Symptom ..."}}

用法（在作业根目录）：
    python baseline/eval_pred.py baseline/out/pred_rule.json
    python baseline/eval_pred.py baseline/out/pred_gated.json --save
    python baseline/eval_pred.py baseline/out/pred_bert.json --tag bert --save

口径定义（报告第⑤节照抄这两条）：

  逐字级 char-level —— 以"字符"为单位比较标签，TP/FP/FN 都是字符数：
      gold!=O 且 pred==gold            -> TP + 1
      gold==O 且 pred!=O               -> FP + 1
      gold!=O 且 pred==O               -> FN + 1
      gold!=O 且 pred!=O 但两者不同    -> FP + 1 且 FN + 1（类型/边界错，双记）
      P=TP/(TP+FP)  R=TP/(TP+FN)  F1=2PR/(P+R)
      acc = 标签完全相同的字符 / 全部字符

  实体级 entity-level —— 先把 BIO 还原成 span=(起, 止, 类型)，
      起、止、类型三者全等才算命中，micro 汇总。

对齐策略：pred 里缺的句子按全 O 处理（即全部漏标，计入 FN），但会大声报告数量，
避免"少交几句反而分数更高"。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(HERE), "bert_api", "code")
sys.path.insert(0, CODE_DIR)

from src.labels import ENTITY_TYPES  # noqa: E402  唯一标签来源，不另起一份

OUT_DIR = os.path.join(HERE, "out")
DEFAULT_GOLD = os.path.join(OUT_DIR, "val_gold.json")


# ---------------------------------------------------------------- 基础工具
def spans(tags):
    """BIO -> [(起, 止, 类型)]，止为开区间。非法 I 开头自动忽略（视为 O）。"""
    out, i, n = [], 0, len(tags)
    while i < n:
        t = tags[i]
        if t.startswith("B-"):
            et = t[2:]
            j = i + 1
            while j < n and tags[j] == "I-" + et:
                j += 1
            out.append((i, j, et))
            i = j
        else:
            i += 1
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def align(gold, pred):
    """对齐 gold/pred，返回 (pairs, report)。pairs: [(did, sid, gold_tags, pred_tags)]"""
    report = {"gold_sentences": 0, "pred_sentences": 0,
              "missing": [], "extra": [], "len_mismatch": []}
    pairs = []
    for did, sents in gold.items():
        for sid, g_str in sents.items():
            report["gold_sentences"] += 1
            g = g_str.split()
            p_str = pred.get(did, {}).get(sid)
            if p_str is None:
                report["missing"].append(f"{did}/{sid}")
                pairs.append((did, sid, g, ["O"] * len(g)))
                continue
            p = p_str.split()
            if len(p) != len(g):
                report["len_mismatch"].append(f"{did}/{sid}: gold={len(g)} pred={len(p)}")
                m = min(len(p), len(g))
                g, p = g[:m], p[:m]
            pairs.append((did, sid, g, p))

    for did, sents in pred.items():
        for sid in sents:
            if did not in gold or sid not in gold.get(did, {}):
                report["extra"].append(f"{did}/{sid}")
    report["pred_sentences"] = sum(len(s) for s in pred.values())
    return pairs, report


# ---------------------------------------------------------------- 逐字级
def char_level(pairs):
    tp = fp = fn = correct = total = 0
    per = {t: [0, 0, 0] for t in ENTITY_TYPES}  # tp, fp, fn

    def add_type(tag, slot):
        et = tag[2:] if "-" in tag else None
        if et in per:
            per[et][slot] += 1

    for _, _, g, p in pairs:
        for a, b in zip(g, p):
            total += 1
            if a == b:
                correct += 1
            if a != "O" and a == b:
                tp += 1
                add_type(a, 0)
            elif a == "O" and b != "O":
                fp += 1
                add_type(b, 1)
            elif a != "O" and b == "O":
                fn += 1
                add_type(a, 2)
            elif a != "O" and b != "O" and a != b:
                fp += 1
                fn += 1
                add_type(b, 1)
                add_type(a, 2)

    p_, r_, f_ = prf(tp, fp, fn)
    per_out = {}
    for t in ENTITY_TYPES:
        a, b, c = per[t]
        pp, rr, ff = prf(a, b, c)
        per_out[t] = {"support": a + c, "tp": a, "fp": b, "fn": c,
                      "P": pp, "R": rr, "F1": ff}
    return {"tp": tp, "fp": fp, "fn": fn,
            "P": p_, "R": r_, "F1": f_,
            "acc": correct / total if total else 0.0,
            "correct": correct, "total": total,
            "per_type": per_out}


# ---------------------------------------------------------------- 实体级
def entity_level(pairs):
    tp = fp = fn = 0
    per = {t: [0, 0, 0] for t in ENTITY_TYPES}
    for _, _, g, p in pairs:
        gs, ps = set(spans(g)), set(spans(p))
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
        for s in gs & ps:
            if s[2] in per:
                per[s[2]][0] += 1
        for s in ps - gs:
            if s[2] in per:
                per[s[2]][1] += 1
        for s in gs - ps:
            if s[2] in per:
                per[s[2]][2] += 1

    p_, r_, f_ = prf(tp, fp, fn)
    per_out = {}
    for t in ENTITY_TYPES:
        a, b, c = per[t]
        pp, rr, ff = prf(a, b, c)
        per_out[t] = {"support": a + c, "tp": a, "fp": b, "fn": c,
                      "P": pp, "R": rr, "F1": ff}
    return {"tp": tp, "fp": fp, "fn": fn, "P": p_, "R": r_, "F1": f_,
            "gold_entities": tp + fn, "pred_entities": tp + fp, "hit": tp,
            "per_type": per_out}


# ---------------------------------------------------------------- 输出
def print_block(title, overall, per_type, unit):
    print(f"\n【{title}】")
    if unit == "char":
        print(f"  字符总数 {overall['total']}   完全正确率 acc={overall['acc']:.4f}")
    else:
        print(f"  标准实体 {overall['gold_entities']}   预测实体 {overall['pred_entities']}"
              f"   命中 {overall['hit']}")
    print(f"  P={overall['P']:.4f}  R={overall['R']:.4f}  F1={overall['F1']:.4f}"
          f"   (tp={overall['tp']} fp={overall['fp']} fn={overall['fn']})")
    print(f"  {'类型':<22}{'支持':>7}{'P':>9}{'R':>9}{'F1':>9}")
    for t in ENTITY_TYPES:
        d = per_type[t]
        print(f"  {t:<22}{d['support']:>7}{d['P']:>9.4f}{d['R']:>9.4f}{d['F1']:>9.4f}")


def main():
    ap = argparse.ArgumentParser(description="统一评测：逐字级 + 实体级 P/R/F1")
    ap.add_argument("pred", help="预测文件，如 baseline/out/pred_rule.json")
    ap.add_argument("--gold", default=DEFAULT_GOLD, help="标准答案，默认 out/val_gold.json")
    ap.add_argument("--tag", default=None, help="结果文件名后缀，默认取 pred 文件名")
    ap.add_argument("--save", action="store_true", help="把汇总写成 out/eval_<tag>.json")
    args = ap.parse_args()

    if not os.path.exists(args.pred):
        print(f"[错误] 预测文件不存在：{args.pred}")
        sys.exit(1)
    if not os.path.exists(args.gold):
        print(f"[错误] 标准文件不存在：{args.gold}")
        sys.exit(1)

    with open(args.pred, encoding="utf-8") as f:
        pred = json.load(f)
    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)

    pairs, rep = align(gold, pred)
    ch = char_level(pairs)
    en = entity_level(pairs)

    name = args.tag or os.path.splitext(os.path.basename(args.pred))[0]
    print("=" * 66)
    print(f"评测：{os.path.basename(args.pred)}")
    print(f"标准：{os.path.basename(args.gold)}")
    print("=" * 66)

    # 对齐检查：任何一项不为 0 都要在报告里说明
    print(f"对齐：gold {rep['gold_sentences']} 句 / pred {rep['pred_sentences']} 句")
    print(f"      缺失 {len(rep['missing'])} 句（按全 O 计）  多余 {len(rep['extra'])} 句"
          f"  标签长度不一致 {len(rep['len_mismatch'])} 句")
    if rep["missing"]:
        print(f"  [警告] 缺失示例：{rep['missing'][:5]}")
    if rep["extra"]:
        print(f"  [警告] 多余示例：{rep['extra'][:5]}")
    if rep["len_mismatch"]:
        print(f"  [警告] 长度不一致示例：{rep['len_mismatch'][:5]}")

    print_block("逐字级 char-level", ch, ch["per_type"], "char")
    print_block("实体级 entity-level", en, en["per_type"], "entity")

    print(f"\nSUMMARY {name} | char P={ch['P']:.4f} R={ch['R']:.4f} F1={ch['F1']:.4f}"
          f" acc={ch['acc']:.4f} | entity P={en['P']:.4f} R={en['R']:.4f} F1={en['F1']:.4f}")

    if args.save:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, f"eval_{name}.json")
        payload = {"pred_file": os.path.abspath(args.pred),
                   "gold_file": os.path.abspath(args.gold),
                   "alignment": {k: (len(v) if isinstance(v, list) else v) for k, v in rep.items()},
                   "char": ch, "entity": en}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"结果已存：{out_path}")


if __name__ == "__main__":
    main()
