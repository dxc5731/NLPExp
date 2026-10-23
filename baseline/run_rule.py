# -*- coding: utf-8 -*-
r"""规则基线在 val 划分上跑分：预测 + 逐字/实体级 P/R/F1 + 错例采样。

用法：
    cd "d:\Research Project\demo\自然语言处理作业"
    $env:IMCS_DATA_DIR="d:\Research Project\demo\自然语言处理作业\IMCS-V2-NER"
    python baseline/run_rule.py                # 默认 min_count=1
    python baseline/run_rule.py 2 min2         # min_count=2，预测另存 pred_rule_min2.json
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(os.path.dirname(HERE), "bert_api", "code")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, HERE)

from rule_baseline import longest_match_bio  # noqa: E402  已废弃，仅兼容旧调用
from fast_match import all_types, build_index, collect_candidates, select_bio  # noqa: E402
from eval_pred import char_level, entity_level, print_block, spans  # noqa: E402  统一口径

ARTIFACTS = os.path.join(CODE_DIR, "artifacts")
DICT_DIR = os.path.join(HERE, "dict")
OUT_DIR = os.path.join(HERE, "out")

ENTITY_TYPES = ["Symptom", "Drug", "Drug_Category", "Medical_Examination", "Operation"]
TYPE2FILE = {
    "Symptom": "dict_symptom.txt",
    "Drug": "dict_drug.txt",
    "Drug_Category": "dict_drug_category.txt",
    "Medical_Examination": "dict_exam.txt",
    "Operation": "dict_operation.txt",
}


def load_lexicon(min_count=1):
    """读 5 个词表，返回 {词: 类型}。同类内同名优先长词，跨类冲突记下来。"""
    lexicon = {}
    conflicts = Counter()
    for etype, fname in TYPE2FILE.items():
        path = os.path.join(DICT_DIR, fname)
        if not os.path.exists(path):
            print(f"[警告] 词表不存在：{path}，请先跑 extract_dict.py")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").rsplit(" ", 1)
                if len(parts) != 2:
                    continue
                word, cnt = parts[0], int(parts[1])
                if cnt < min_count or not word:
                    continue
                if word in lexicon and lexicon[word] != etype:
                    conflicts[word] = 1
                    continue
                lexicon[word] = etype
    return lexicon, conflicts


def load_split_ids():
    with open(os.path.join(ARTIFACTS, "split_v1.json"), encoding="utf-8") as f:
        return json.load(f)


def load_examples(dialogue_ids, split_name="train"):
    from src.data_io import load_split
    from src.dataset import collect_examples

    data = load_split(split_name)
    ids = [i for i in dialogue_ids if i in data]
    return collect_examples(data, ids)


def run_val(min_count=1, show=10, tag=None):
    lexicon, conflicts = load_lexicon(min_count=min_count)
    print(f"词表载入：{len(lexicon)} 词" + (f"（跨类冲突丢弃 {len(conflicts)} 个）" if conflicts else ""))

    split = load_split_ids()
    val_ids = split["val"]
    examples = load_examples(val_ids)
    print(f"val 对话数={len(val_ids)}  句子数={len(examples)}\n")

    gold_all, pred_all, pred_json = [], [], {}
    samples = []
    # 一次性建索引 + 一次扫描候选；语义与 longest_match_bio 等价（verify_fast_match.py 已逐句核验），
    # 但把"每句都把 8354 词表重排一遍"的 O(句数 × 词表) 开销降下来，全量 val 从分钟级降到秒级。
    index = build_index(lexicon)
    allow = all_types(index)
    for ex in examples:
        text, tags = ex["text"], ex["tags"]
        pred = select_bio(text, collect_candidates(text, index), allow)
        assert len(pred) == len(text) == len(tags), (ex["dialogue_id"], ex["sentence_id"])
        gold_all.append(tags)
        pred_all.append(pred)
        pred_json.setdefault(ex["dialogue_id"], {})[ex["sentence_id"]] = " ".join(pred)
        if len(samples) < show:
            samples.append((text, tags, pred))

    os.makedirs(OUT_DIR, exist_ok=True)
    # 带 tag 时另存一份，避免参数扫描互相覆盖（pred_rule_min2.json 等）
    pred_name = "pred_rule.json" if not tag else f"pred_rule_{tag}.json"
    pred_path = os.path.join(OUT_DIR, pred_name)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_json, f, ensure_ascii=False)

    gold_path = os.path.join(OUT_DIR, "val_gold.json")
    with open(gold_path, "w", encoding="utf-8") as f:
        json.dump(pred_json and _to_gold_json(examples), f, ensure_ascii=False)

    pairs = [(ex["dialogue_id"], ex["sentence_id"], g, p)
             for ex, g, p in zip(examples, gold_all, pred_all)]
    ch = char_level(pairs)
    en = entity_level(pairs)
    print("=" * 62)
    print("指标（统一口径，口径定义见 baseline/eval_pred.py 文件头）")
    print("=" * 62)
    print_block("逐字级 char-level", ch, ch["per_type"], "char")
    print_block("实体级 entity-level", en, en["per_type"], "entity")
    m = {"token_acc": ch["acc"], "P": ch["P"], "R": ch["R"], "F1": ch["F1"]}

    print(f"\n预测文件：{pred_path}")
    print(f"标准文件：{gold_path}")

    print("\n" + "=" * 62)
    print(f"肉眼抽查（前 {len(samples)} 句）")
    print("=" * 62)
    for text, g, p in samples:
        print(f"\n原文: {text}")
        print(f"标准: {' '.join(g)}")
        print(f"预测: {' '.join(p)}")

    dump_error_samples(examples, gold_all, pred_all, lexicon)
    return m


def dump_error_samples(examples, gold_all, pred_all, lexicon, per_type=3):
    """【已废弃】错例分析已重写为 baseline/error_analysis.py。

    旧实现有三个问题，报告里不要再引用它的数字：
      1. `elif` 与 `else` 两个分支代码完全相同（死代码），"误标"桶什么都往里塞；
      2. "边界错"的判据是"同词同类出现在别处"，那是位置错，不是边界错；
      3. 只有 3 个桶且没有归因，不满足任务书"≥3 类失败并逐一归因"的要求。

    新实现：python baseline/error_analysis.py
    —— 类别互斥（边界错/类型错/漏标/误标），每类带子类、数量、占比、top 词，
       并单列金标准内部 span 重叠（数据噪声）。
    """
    print("\n[提示] 错例分析请改用：python baseline/error_analysis.py")
    return {}


def _to_gold_json(examples):
    out = {}
    for ex in examples:
        out.setdefault(ex["dialogue_id"], {})[ex["sentence_id"]] = " ".join(ex["tags"])
    return out


def _spans(tags):
    """【兼容旧调用】与 eval_pred.spans 共用同一份实现，防止两处 BIO 解析走样。"""
    return spans(tags)


def _per_type_report(gold_all, pred_all):
    """【已废弃，保留仅为兼容旧调用】改用 eval_pred.entity_level / char_level。

    原实现有个隐蔽 bug：`any(gs_ == (s, e, t) for gs_, _, t in gs)` 里
    gs_ 是起始下标 int，却拿去和三元组比较，永远为 False，导致每个预测
    span 都被当成 FP —— per-type 的 P 被严重低估（Symptom 0.361 实际 0.565）。
    报告里的逐类数字一律用 eval_pred.py 的输出。
    """
    raise NotImplementedError("请改用 eval_pred.entity_level / char_level")


if __name__ == "__main__":
    mc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    tag = sys.argv[2] if len(sys.argv) > 2 else None
    run_val(min_count=mc, tag=tag)