# -*- coding: utf-8 -*-
"""统计 dev 中「发热」是否被标注：正确口径是按区间重叠判断，而非字符串完全相等。"""
from src.data_io import load_split


def spans(tags):
    out, i = [], 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            e, j = tags[i][2:], i + 1
            while j < len(tags) and tags[j] == "I-" + e:
                j += 1
            out.append((i, j, e))
            i = j
        else:
            i += 1
    return out


def main() -> None:
    dv = load_split("dev")
    covered = 0          # 「发热」整体落在某个金标实体内部
    overlap = 0          # 「发热」与某个实体有部分重叠
    none = 0             # 完全不重叠：真的没标
    none_list = []
    for did, d in dv.items():
        for s in d["dialogue"]:
            t = s["sentence"]
            g = spans(s["BIO_label"].split(" "))
            start = t.find("发热")
            while start != -1:
                end = start + 2
                inside = [x for x in g if x[0] <= start and end <= x[1]]
                part = [x for x in g if max(0, min(end, x[1]) - max(start, x[0])) > 0]
                if inside:
                    covered += 1
                elif part:
                    overlap += 1
                else:
                    none += 1
                    none_list.append((did, s["sentence_id"], t))
                start = t.find("发热", start + 1)

    total = covered + overlap + none
    print(f"dev 中「发热」共出现 {total} 次")
    print(f"  被完整包含在某个实体里：{covered}")
    print(f"  与实体部分重叠：{overlap}")
    print(f"  完全没有被标注：{none}  ({none/total:.1%})")
    print("\n未被标注的句子：")
    for did, sid, t in none_list[:20]:
        print(f"  [{did}/{sid}] {t[:60]}")


if __name__ == "__main__":
    main()
