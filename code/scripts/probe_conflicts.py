# -*- coding: utf-8 -*-
"""把这 8 句里出问题的词片单独拎出来看：它跨了几个字符、这些字符的金标分别是什么。"""
from src.align import encode_sentence, load_tokenizer
from src.data_io import load_split

tok = load_tokenizer()
data = load_split("train")
bad = [("10034734", "2"), ("10098865", "21"), ("10193361", "25"), ("10206772", "25"),
       ("10491432", "7"), ("10045298", "9")]
for did, sid in bad:
    for s in data[did]["dialogue"]:
        if s["sentence_id"] != sid:
            continue
        text = s["sentence"]
        tags = s["BIO_label"].split(" ")
        ids, offs = encode_sentence(tok, text)
        print(f"--- {did}/{sid} ---")
        print(text)
        for (st, en) in offs:
            if en - st > 1:
                inner = [(k, tags[k]) for k in range(st, en)]
                print(f"  词片 {text[st:en]!r} 覆盖 {st}-{en}：{inner}")
        print()
