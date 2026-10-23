# -*- coding: utf-8 -*-
"""错例分析：把规则基线在 val 上的错误拆成**互斥的类别**，每类给出数量、占比和归因。

为什么要重写（旧的 dump_error_samples 有三个问题）：
  1. `elif` 和 `else` 两个分支代码完全一样 —— 死代码，导致"误标"桶什么都往里塞；
  2. "边界错"的判据是"同词同类出现在别的位置"，那其实是**位置错**；
     真正的边界错是"长实体被词典里的短词截断"，判据完全不同；
  3. 只分 3 桶，不足以支撑任务书要求的"≥3 类失败并逐一归因"。

本脚本的分类（互斥，一条错误只归一类，判据按顺序短路）：

  ① 边界错（截断/多切）
     预测 span 与某个金标准 span **有重叠**但边界不同。子情况：
       - 截断：预测是金标准的真子串（如金标准"一天拉6到7次"，只标了"拉"）
       - 扩界：预测把金标准包在里面（多标了字）
       - 错位：两者部分重叠但都不是对方的子串

  ② 类型错（判错类）
     预测 span 的字面内容**与某个金标准 span 完全相同**，但类型不同。
     例：金标准把"头孢"标成 Drug，规则把它标成 Drug_Category。

  ③ 漏标
     金标准 span 完全没被预测覆盖。
     本分支能到达的一定是**词表外未登录**——若该词在词典里，最长匹配必然命中它；
     没命中只可能是被更长的词吃掉，而那属于重叠，已归入①边界错，不重复计。

  ④ 误标（凭空多标）
     预测 span 与所有金标准 span **都不重叠**，且字串不在金标准里。
     子情况：
       - 高频歧义词：常见口语短词在别的语境里根本不是实体（如"拉""胀""烧"）
       - 短语误切：把一个非实体的固定搭配切成了实体（如"查过大便"）

  ⑤ 嵌套冲突（标注本身歧义）
     金标准里两个 span 互相重叠（本数据集的 BIO 标注理论上不允许，
     真出现了说明是数据噪声），单独统计，作为**数据质量**指标而非模型错误。

每一类的 top 词频也一并给出，报告里可以直接引用。

用法：
    python baseline/error_analysis.py            # 默认 min_count=1
    python baseline/error_analysis.py 10         # 用 min_count=10 的词典

产出：
    out/errors_sample.txt      文本版错例（每类若干例，带归因）
    out/error_analysis.json    机器可读（各类计数、占比、top 词）
    out/error_analysis_log.txt 统计摘要（由调用方重定向）
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(HERE), "bert_api", "code")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, HERE)

from eval_pred import spans  # noqa: E402
from fast_match import all_types, build_index, collect_candidates, select_bio  # noqa: E402
from run_rule import load_examples, load_lexicon, load_split_ids  # noqa: E402

OUT = os.path.join(HERE, "out")


def overlap(a, b):
    """两个 span [s,e) 是否重叠。"""
    return a[0] < b[1] and b[0] < a[1]


def classify_sentence(text, gold_tags, pred_tags, lexicon):
    """把一句里的错误分类。返回 (错误列表, 统计增量 dict)。

    错误项格式：(类别, 子类, 预测span, 金标准span或None, 字串)
    """
    gs = spans(gold_tags)
    ps = spans(pred_tags)
    gset = {(s, e, t) for s, e, t in gs}
    pset = {(s, e, t) for s, e, t in ps}
    errs = []

    def gold_at(s, e):
        """与 [s,e) 有重叠的所有金标准 span。"""
        return [x for x in gs if overlap((s, e), x)]

    # ---- 预测侧错误 ----
    for (s, e, t) in ps:
        if (s, e, t) in gset:
            continue
        cand = gold_at(s, e)
        if cand:
            # 有重叠但不完全相同 -> 边界类
            best = min(cand, key=lambda x: (abs(x[0] - s) + abs(x[1] - e)))
            g0, g1, gt = best
            if s == g0 and e < g1:
                sub = "截断"
            elif s > g0 and e == g1:
                sub = "扩展"
            else:
                sub = "错位"
            errs.append(("边界错", sub, (s, e, t), best, text[s:e]))
        else:
            # 无重叠：可能是类型错（同字面不同类）或纯误标
            same_text = [x for x in gs if text[x[0]:x[1]] == text[s:e]]
            if same_text:
                errs.append(("类型错", "同字面异类型", (s, e, t), same_text[0], text[s:e]))
            else:
                errs.append(("误标", "非实体被标", (s, e, t), None, text[s:e]))

    # ---- 金标准侧未被覆盖 -> 漏标 ----
    for (s, e, t) in gs:
        if (s, e, t) in pset:
            continue
        if any(overlap((s, e), p) for p in ps):
            continue  # 已被"边界错"覆盖，不重复计
        word = text[s:e]
        # 到这里说明这个词既没被预测覆盖、也不和任何预测重叠。
        # 若它在词典里，正常来说最长匹配一定会命中它；没命中只可能是
        # 它被一个更长的词"吃掉"了——但那属于重叠，已在上面被划走。
        # 因此本分支能到达的，全都是**词典里根本没有的词**（未登录词）。
        errs.append(("漏标", "词表外未登录", None, (s, e, t), word))

    return errs


def detect_nested(gold_tags):
    """金标准内部 span 重叠 -> 标注歧义/噪声。返回 [(span1, span2)]。"""
    gs = spans(gold_tags)
    out = []
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            if overlap(gs[i], gs[j]):
                out.append((gs[i], gs[j]))
    return out


def main(min_count=1, show=5):
    lexicon, conflicts = load_lexicon(min_count=min_count)
    split = load_split_ids()
    examples = load_examples(split["val"])
    print(f"词典 min_count={min_count}：{len(lexicon)} 词（跨类冲突丢弃 {len(conflicts)}）")
    print(f"val {len(examples)} 句\n")

    index = build_index(lexicon)
    allow = all_types(index)

    buckets = {}          # (类别, 子类) -> [错误项]
    nested_total = 0
    nested_examples = []
    gold_ent_total = 0
    pred_ent_total = 0

    for ex in examples:
        text = ex["text"]
        pred = select_bio(text, collect_candidates(text, index), allow)
        gold = ex["tags"]
        gs, ps = spans(gold), spans(pred)
        gold_ent_total += len(gs)
        pred_ent_total += len(ps)

        nest = detect_nested(gold)
        if nest:
            nested_total += len(nest)
            if len(nested_examples) < show:
                nested_examples.append((ex, nest))

        for kind, sub, pspan, gspan, word in classify_sentence(text, gold, pred, lexicon):
            buckets.setdefault((kind, sub), []).append((ex, pspan, gspan, word))

    # ---- 统计 ----
    total_err = sum(len(v) for v in buckets.values())
    print(f"金标准实体 {gold_ent_total} 个 / 预测实体 {pred_ent_total} 个")
    print(f"错误总数（按 span 计）= {total_err}")
    print(f"金标准内部 span 重叠（标注歧义）= {nested_total}\n")

    by_kind = Counter()
    for (kind, sub), items in buckets.items():
        by_kind[kind] += len(items)
    print(f"{'类别':<10}{'子类':<16}{'数量':>8}{'占比':>9}   top 词")
    print("-" * 78)
    for kind in ["边界错", "类型错", "漏标", "误标"]:
        for (k, sub), items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            if k != kind:
                continue
            top = Counter(w for _, _, _, w in items).most_common(6)
            print(f"{k:<10}{sub:<16}{len(items):>8}{len(items) / max(1, total_err):>9.1%}"
                  f"   {'、'.join(w for w, _ in top)}")
    print("-" * 78)
    print(f"合计 {total_err}   另：金标准内部重叠 {nested_total}（属数据噪声，单列）")

    # ---- 写文本错例 ----
    os.makedirs(OUT, exist_ok=True)
    sfx = "" if min_count == 1 else f"_mc{min_count}"
    path = os.path.join(OUT, f"errors_sample{sfx}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 规则基线错例分析  词典 min_count={min_count}（{len(lexicon)} 词）\n")
        f.write(f"# val {len(examples)} 句，错误 span 合计 {total_err} 个\n")
        f.write("# 类别口径：① 边界错（截断/扩展/错位） ② 类型错（同字面异类型）"
                " ③ 漏标（词表外未登录 / 词表内被抢） ④ 误标（凭空多标）\n\n")
        for (kind, sub), items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            f.write(f"\n{'=' * 72}\n【{kind} / {sub}】共 {len(items)} 例，"
                    f"占全部错误 {len(items) / max(1, total_err):.1%}，展示前 {show} 例\n{'=' * 72}\n")
            for ex, pspan, gspan, word in items[:show]:
                text = ex["text"]
                pred = select_bio(text, collect_candidates(text, index), allow)
                f.write(f"对话 {ex['dialogue_id']} 句 {ex['sentence_id']}\n")
                f.write(f"原文: {text}\n")
                f.write(f"标注: {' '.join(ex['tags'])}\n")
                f.write(f"预测: {' '.join(pred)}\n")
                f.write(f"问题: 「{word}」")
                if gspan is not None:
                    f.write(f"  金标准 span=({gspan[0]},{gspan[1]}) 类型={gspan[2]}")
                if pspan is not None:
                    f.write(f"  预测 span=({pspan[0]},{pspan[1]}) 类型={pspan[2]}")
                f.write(f"  → {kind}/{sub}\n\n")
        if nested_examples:
            f.write(f"\n{'=' * 72}\n【数据噪声】金标准内部 span 重叠 {nested_total} 例，展示前 {show} 例\n{'=' * 72}\n")
            for ex, nest in nested_examples:
                f.write(f"对话 {ex['dialogue_id']} 句 {ex['sentence_id']}\n")
                f.write(f"原文: {ex['text']}\n")
                f.write(f"标注: {' '.join(ex['tags'])}\n")
                for a, b in nest:
                    f.write(f"  重叠: ({a[0]},{a[1]}){a[2]}「{ex['text'][a[0]:a[1]]}」"
                            f"  与  ({b[0]},{b[1]}){b[2]}「{ex['text'][b[0]:b[1]]}」\n")
                f.write("\n")
    print(f"\n错例已写：{path}")

    # ---- 写机器可读 ----
    payload = {
        "min_count": min_count,
        "n_lexicon": len(lexicon),
        "n_val_sentences": len(examples),
        "gold_entities": gold_ent_total,
        "pred_entities": pred_ent_total,
        "total_errors": total_err,
        "nested_gold_overlaps": nested_total,
        "by_category": {
            kind: sum(len(v) for (k, _), v in buckets.items() if k == kind)
            for kind in ["边界错", "类型错", "漏标", "误标"]
        },
        "by_subcategory": {
            f"{k}/{sub}": {
                "count": len(items),
                "ratio": len(items) / max(1, total_err),
                "top_words": Counter(w for _, _, _, w in items).most_common(20),
            }
            for (k, sub), items in buckets.items()
        },
    }
    with open(os.path.join(OUT, f"error_analysis{sfx}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"统计已写：{os.path.join(OUT, f'error_analysis{sfx}.json')}")
    print(f"SUMMARY error_analysis min_count={min_count} total_errors={total_err}"
          f" by_category={payload['by_category']} nested={nested_total}")


if __name__ == "__main__":
    mc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(min_count=mc)
