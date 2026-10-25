# -*- coding: utf-8 -*-
"""按《预测结果交接规范》独立校验一份预测文件。

这不是组长那份校验器的替代品（那份我们没拿到），而是交之前自己先把能查的全查一遍。

用法（在 code/ 目录下）：
    python -X utf8 -m scripts.validate_predictions --predictions <路径> --split dev

检查项：
  1. UTF-8 无 BOM、严格 JSON、无重复 key
  2. 两层结构：对话ID -> 句子ID -> BIO 字符串
  3. 对话/句子 ID 集合与原始数据完全一致，不多不少不改名
  4. 每个原始字符一个标签，数量 == len(sentence)
  5. 只用 11 种规定标签
  6. 单个 ASCII 空格分隔，无首尾空格、连续空格、制表符、换行
  7. 每句 BIO 合法（I-X 前面必须是同句的 B-X 或 I-X）
"""

import argparse
import json
import sys

from src.data_io import load_split
from src.labels import LABELS

MAX_EXAMPLES = 5


def load_json_strict(raw: bytes):
    dup = []

    def hook(pairs):
        seen = set()
        for k, _ in pairs:
            if k in seen:
                dup.append(k)
            seen.add(k)
        return dict(pairs)

    return json.loads(raw.decode("utf-8"), object_pairs_hook=hook), dup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    errors = []
    def fail(code, detail):
        errors.append((code, detail))

    raw = open(args.predictions, "rb").read()
    if raw[:3] == b"\xef\xbb\xbf":
        fail("BOM", "文件带 UTF-8 BOM")
    try:
        pred, dup = load_json_strict(raw)
    except Exception as e:
        print(f"FAIL JSON 解析失败: {e}")
        sys.exit(1)
    if dup:
        fail("DUPLICATE_KEY", f"{len(dup)} 个重复 key，例如 {dup[:3]}")

    if not isinstance(pred, dict):
        print("FAIL 根节点不是对象")
        sys.exit(1)

    data = load_split(args.split)
    if set(pred) != set(data):
        missing = set(data) - set(pred)
        extra = set(pred) - set(data)
        fail("DIALOGUE_ID", f"缺 {len(missing)} 个对话，多 {len(extra)} 个")

    n_sent = 0
    n_char = 0
    bad_len = bad_tag = bad_space = bad_bio = missing_sent = 0

    for did, d in data.items():
        if did not in pred:
            continue
        gold_sids = [s["sentence_id"] for s in d["dialogue"]]
        if set(pred[did]) != set(gold_sids):
            missing_sent += 1
        for s in d["dialogue"]:
            sid = s["sentence_id"]
            if sid not in pred[did]:
                continue
            n_sent += 1
            text = s["sentence"]
            n_char += len(text)
            value = pred[did][sid]
            if not isinstance(value, str):
                bad_tag += 1
                continue
            if value != value.strip() or "  " in value or "\t" in value or "\n" in value:
                bad_space += 1
            tags = value.split(" ")
            if len(tags) != len(text):
                bad_len += 1
                if bad_len <= MAX_EXAMPLES:
                    fail("LENGTH", f"{did}/{sid}: {len(tags)} 标签 vs {len(text)} 字符")
                continue
            if any(t not in LABELS for t in tags):
                bad_tag += 1
                continue
            prev = "O"
            for i, t in enumerate(tags):
                if t.startswith("I-"):
                    ent = t[2:]
                    if prev not in ("B-" + ent, "I-" + ent):
                        bad_bio += 1
                        if bad_bio <= MAX_EXAMPLES:
                            fail("BIO", f"{did}/{sid} 第 {i} 个字符: {prev} -> {t}")
                        break
                prev = t

    print("=" * 60)
    print(f"文件 {args.predictions}")
    print(f"划分 {args.split}：原始 {len(data)} 段对话，预测 {len(pred)} 段对话")
    print(f"句子 {n_sent}，字符 {n_char}")
    print(f"长度不一致 {bad_len}，非法标签 {bad_tag}，空格问题 {bad_space}，"
          f"非法 BIO {bad_bio}，句子缺漏 {missing_sent}")
    for code, detail in errors[:MAX_EXAMPLES]:
        print(f"  [{code}] {detail}")
    print("=" * 60)
    if errors or bad_len or bad_tag or bad_space or bad_bio or missing_sent:
        print("FAIL")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
