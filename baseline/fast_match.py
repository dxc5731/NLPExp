# -*- coding: utf-8 -*-
"""规则匹配加速版：语义与 `rule_baseline.longest_match_bio` 完全一致，只是把
"每次调用都把整个词表排序一遍" 换成 "一次建索引，反复复用"。

为什么需要它：原实现每次调用都 `sorted(lexicon.items(), key=-len)`；
val 有 9687 句、词表 8354 词，等于排序 9687 次，一次全量要几分钟。
参数扫描（min_count × θ）要跑几十次，原实现根本跑不动。

等价性由 `baseline/verify_fast_match.py` 逐句比对验证，不要凭感觉。
"""
from label_map import check_bio


def build_index(lexicon):
    """{首字: [(词, 类型, 入表次序), ...]}，组内按 (词长降序, 入表次序) 排。

    保留"入表次序"是为了复刻原实的平局规则：原实现按 -len 做**稳定**排序，
    同长度时先入表者优先；索引里带上 rank 才能保证同长度同起点时不走样。
    """
    idx = {}
    for rank, (w, t) in enumerate(lexicon.items()):
        if not w:
            continue
        idx.setdefault(w[0], []).append((w, t, rank))
    for k in idx:
        idx[k].sort(key=lambda x: (-len(x[0]), x[2]))
    return idx


def collect_candidates(text, index):
    """扫一遍，记下每个起下标的全部命中词（已按词长降序）。

    返回 {起下标: [(词, 类型, rank), ...]}，只含真正在 text 中出现的词。
    有了它，换一个"本句有哪些类"的组合就不用重新扫描文本了。
    """
    hits = {}
    for i, ch in enumerate(text):
        lst = index.get(ch)
        if not lst:
            continue
        found = [item for item in lst if text.startswith(item[0], i)]
        if found:
            hits[i] = found
    return hits


def select_bio(text, hits, present_types):
    """在"本句只允许这些类"的前提下做最长匹配。

    等价于：把词表过滤成 present_types 子表，再跑原版最长匹配。
    """
    n = len(text)
    tags = ["O"] * n
    i = 0
    while i < n:
        pick = None
        for w, t, _ in hits.get(i, ()):
            if t in present_types:
                pick = (w, t)
                break
        if pick:
            w, t = pick
            tags[i] = "B-" + t
            for k in range(1, len(w)):
                tags[i + k] = "I-" + t
            i += len(w)
        else:
            i += 1
    return check_bio(tags)


def match_one(text, index, present_types):
    """单句便捷入口（内部收集一次候选）。present_types 传 None 表示不限类。"""
    text = (text or "").strip()
    hits = collect_candidates(text, index)
    if present_types is None:
        present_types = all_types(index)
    return select_bio(text, hits, present_types)


def all_types(index):
    return {t for lst in index.values() for _, t, _ in lst}
