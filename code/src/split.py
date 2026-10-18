# -*- coding: utf-8 -*-
"""按「整段对话」切分官方 train，产出全组共用的内部 train / val 名单。

为什么要按对话切：
    同一段对话里前后句子高度相似（同一个病人、同一个医生、同一套说法）。
    如果按句子随机切，几乎一样的句子会同时出现在训练集和验证集里，
    验证分数会虚高，而且虚高得很隐蔽——不报错，只是数字好看。

为什么固定种子：
    切分结果必须可复现，也必须让组里每个人拿到的名单完全一致。
    排序后再打乱，是为了不依赖 JSON 里键的先后顺序。

用法（在 code/ 目录下执行）：
    python -m src.split
"""

import json
import random
from pathlib import Path

from .config import ARTIFACTS, DATA_DIR
from .data_io import load_split, sha256_file
from .labels import ENTITY_TYPES

SEED = 42
VAL_RATIO = 0.10
VERSION = "v1"
OUT_DIR = ARTIFACTS


def dialogue_stats(dialogue: dict) -> dict:
    """一段对话的规模：句子数、字符数、五类实体各多少个。"""
    n_sent = len(dialogue["dialogue"])
    n_char = sum(len(s["sentence"]) for s in dialogue["dialogue"])
    ents = {t: 0 for t in ENTITY_TYPES}
    for s in dialogue["dialogue"]:
        for tag in s["BIO_label"].split(" "):
            if tag.startswith("B-"):
                ents[tag[2:]] += 1
    return {"sentences": n_sent, "chars": n_char, "entities": ents}


def make_split(ids: list, seed: int = SEED, val_ratio: float = VAL_RATIO):
    rng = random.Random(seed)
    shuffled = sorted(ids)          # 先排序，消除输入顺序的影响
    rng.shuffle(shuffled)           # 再打乱，种子固定则结果固定
    n_val = int(round(len(shuffled) * val_ratio))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


def main() -> None:
    data = load_split("train")
    ids = list(data.keys())
    train_ids, val_ids = make_split(ids)

    # ---- 自检：这三条如果不过，后面所有实验都作废 ----
    assert set(train_ids).isdisjoint(val_ids), "训练集和验证集有重叠对话"
    assert set(train_ids) | set(val_ids) == set(ids), "有对话没被分到任何一边"
    assert len(train_ids) + len(val_ids) == len(ids)
    # 官方 dev 绝不参与切分；这里只是再确认一次没被混进来
    assert len(load_split("dev")) == 833

    agg = {"train": {"dialogues": 0, "sentences": 0, "chars": 0,
                     "entities": {t: 0 for t in ENTITY_TYPES}},
           "val": {"dialogues": 0, "sentences": 0, "chars": 0,
                   "entities": {t: 0 for t in ENTITY_TYPES}}}
    for name, split_ids in (("train", train_ids), ("val", val_ids)):
        for did in split_ids:
            st = dialogue_stats(data[did])
            agg[name]["dialogues"] += 1
            agg[name]["sentences"] += st["sentences"]
            agg[name]["chars"] += st["chars"]
            for t in ENTITY_TYPES:
                agg[name]["entities"][t] += st["entities"][t]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VERSION,
        "created_by": "member_A",
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "source_file": "IMCS-V2_train.json",
        "source_sha256": sha256_file(DATA_DIR / "IMCS-V2_train.json"),
        "n_dialogues": len(ids),
        "train": train_ids,
        "val": val_ids,
        "stats": agg,
    }
    out = OUT_DIR / f"split_{VERSION}.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"名单写入 {out}")
    print(f"源文件 SHA-256 {payload['source_sha256']}")
    for name in ("train", "val"):
        a = agg[name]
        ent = "  ".join(f"{t}={a['entities'][t]}" for t in ENTITY_TYPES)
        print(f"{name:5s} 对话={a['dialogues']:5d} 句={a['sentences']:6d} "
              f"字符={a['chars']:8d}  {ent}")


if __name__ == "__main__":
    main()
