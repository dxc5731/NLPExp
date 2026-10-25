"""规则基线：最长匹配扫字 + BIO 合法性。复用 DEMO turn_analyzer 归一化思想。"""
from label_map import check_bio

def normalize(text: str) -> str:
    return (text or "").strip()

def longest_match_bio(text, lexicon):
    """text: str; lexicon: {词: 类别}; 返回 List[str] BIO，与字一一对应。"""
    text = normalize(text)
    n = len(text)
    tags = ["O"] * n
    i = 0
    # 按词长降序，最长优先
    words = sorted(lexicon.items(), key=lambda kv: -len(kv[0]))
    while i < n:
        hit = None
        for w, typ in words:
            if text.startswith(w, i):
                hit = (w, typ)
                break
        if hit:
            w, typ = hit
            tags[i] = "B-" + typ
            for k in range(1, len(w)):
                tags[i + k] = "I-" + typ
            i += len(w)
        else:
            i += 1
    return check_bio(tags)

def token_prf(gold, pred):
    """逐字 P/R/F1。gold/pred: List[List[str]]"""
    tp = fp = fn = 0
    tp_tok = ok = tot = 0
    for g, p in zip(gold, pred):
        for a, b in zip(g, p):
            tot += 1
            if a == b:
                ok += 1
            if a != "O" and a == b:
                tp += 1
            elif a == "O" and b != "O":
                fp += 1
            elif a != "O" and b == "O":
                fn += 1
            elif a != "O" and b != "O" and a != b:
                fp += 1; fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"token_acc": ok / tot if tot else 0.0, "P": prec, "R": rec, "F1": f1}
