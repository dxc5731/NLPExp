# -*- coding: utf-8 -*-
"""找出覆盖超过 8 个字符的词片，看看它们到底是什么。"""
from src.align import encode_sentence, load_tokenizer
from src.data_io import load_split

tok = load_tokenizer()
for split in ("train", "dev", "test"):
    data = load_split(split)
    for did, d in data.items():
        for s in d["dialogue"]:
            text = s["sentence"]
            ids, offs = encode_sentence(tok, text)
            for i, (st, en) in enumerate(offs):
                if en - st > 8:
                    print(f"[{split}] 对话{did} 句{s['sentence_id']} "
                          f"跨度{en - st} 原文片段={text[st:en]!r} "
                          f"词片={tok.convert_ids_to_tokens([ids[i]])}")
                    print("   整句:", text)
