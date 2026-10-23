# -*- coding: utf-8 -*-
"""TF-IDF 小模型基线（纯 numpy 实现，避开环境里 sklearn/NumPy2 冲突）。

char 2-4gram TF-IDF + 每类 Rocchio 中心向量分类。
这是任务书 B 行要求的“规则/词典 + 小模型”里的 +小模型 位。
"""
import math
from collections import Counter


def char_ngrams(text, n_min=2, n_max=4):
    out = []
    for n in range(n_min, n_max + 1):
        for i in range(max(0, len(text) - n + 1)):
            out.append(text[i:i + n])
    return out


def build_vocab(texts, max_features=20000):
    df = Counter()
    for t in texts:
        for g in set(char_ngrams(t)):
            df[g] += 1
    n = len(texts)
    idf = {}
    for g, c in df.items():
        idf[g] = math.log((n + 1) / (c + 1)) + 1.0
    vocab = [g for g, _ in sorted(df.items(), key=lambda kv: -kv[1])[:max_features]]
    return {g: i for i, g in enumerate(vocab)}, idf


def tfidf_row(text, vocab, idf):
    tf = Counter(char_ngrams(text))
    total = sum(tf.values()) or 1
    vec = {}
    for g, c in tf.items():
        if g in vocab:
            vec[vocab[g]] = (c / total) * idf[g]
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def train_ovr(X_texts, Y):
    """Y: List[List[int]] 5 维 0/1。返回 (vocab, idf, centers)，centers[j] 为正类中心向量。"""
    vocab, idf = build_vocab(X_texts)
    dim = len(vocab)
    centers = []
    for j in range(len(Y[0])):
        acc = [0.0] * dim
        cnt = 0
        for x, y in zip(X_texts, Y):
            if y[j] == 1:
                cnt += 1
                for k, v in tfidf_row(x, vocab, idf).items():
                    acc[k] += v
        if cnt:
            acc = [v / cnt for v in acc]
        norm = math.sqrt(sum(v * v for v in acc)) or 1.0
        centers.append([v / norm for v in acc])
    return vocab, idf, centers


def _cos(a, b_vec):
    s = sum(v * b_vec[k] for k, v in a.items())
    return s  # 两边都已归一化


def predict(vocab, idf, centers, X_texts, thresh=0.05):
    out = []
    for x in X_texts:
        v = tfidf_row(x, vocab, idf)
        out.append([1 if _cos(v, c) >= thresh else 0 for c in centers])
    return out
