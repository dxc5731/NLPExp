# -*- coding: utf-8 -*-
"""对齐层的自动测试。

跑法（在 code/ 目录下）：
    python -m unittest discover -s tests -v

这些测试回答的是「对齐会不会静默出错」：
错位不会报错，只会让 F1 莫名其妙地低，所以必须用断言钉死。
"""

import unittest

from src.align import (
    IGNORE_INDEX,
    encode_sentence,
    gold_to_token_labels,
    load_tokenizer,
    tokens_to_char_labels,
)
from src.dataset import collate_fn
from src.labels import LABEL2ID


class AlignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = load_tokenizer()

    def test_offsets_cover_every_char_once(self):
        text = "孩子发烧38.5度，吃了小儿吗丁啉"
        ids, offsets = encode_sentence(self.tok, text)
        covered = [0] * len(text)
        for s, e in offsets:
            for k in range(s, e):
                covered[k] += 1
        self.assertEqual(covered, [1] * len(text), "有字符没被覆盖，或被覆盖多次")

    def test_special_tokens_get_ignore_index(self):
        text = "您好"
        ids, offsets = encode_sentence(self.tok, text)
        labels = gold_to_token_labels(["O"] * len(text), offsets)
        self.assertEqual(labels[0], IGNORE_INDEX)      # [CLS]
        self.assertEqual(labels[-1], IGNORE_INDEX)     # [SEP]
        self.assertEqual(len(labels), len(ids))

    def test_multichar_token_expands_b_to_i(self):
        """一个词片覆盖两个字符时，第二个字符不能也是 B，否则实体被截成两个。"""
        text = "38"
        ids, offsets = encode_sentence(self.tok, text)
        labels = [IGNORE_INDEX] * len(ids)
        for i, (s, e) in enumerate(offsets):
            if s != e:
                labels[i] = LABEL2ID["B-Symptom"]
        restored, uncovered = tokens_to_char_labels(labels, offsets, len(text))
        self.assertEqual(restored, ["B-Symptom", "I-Symptom"])
        self.assertEqual(uncovered, [])

    def test_leading_i_is_repaired_to_b(self):
        """句首出现 I-X 是非法 BIO，还原时必须改写成 B-X。"""
        restored, _ = tokens_to_char_labels(
            [LABEL2ID["I-Symptom"]], [(0, 3)], 3
        )
        self.assertEqual(restored[0], "B-Symptom")
        self.assertEqual(restored[1:], ["I-Symptom", "I-Symptom"])

    def test_roundtrip_on_plain_sentence(self):
        text = "但不能祛痰，所以建议加上氨溴索嘛"
        tags = ("O O O O B-Symptom O O O O O O O "
                "B-Drug I-Drug I-Drug O").split(" ")
        self.assertEqual(len(tags), len(text))
        ids, offsets = encode_sentence(self.tok, text)
        token_labels = gold_to_token_labels(tags, offsets)
        restored, _ = tokens_to_char_labels(token_labels, offsets, len(text))
        self.assertEqual(restored, tags)

    def test_collate_pads_with_ignore_index(self):
        ids1, off1 = encode_sentence(self.tok, "你好")
        ids2, off2 = encode_sentence(self.tok, "你好，请问孩子发烧几天了")
        batch = [
            {"input_ids": ids1, "attention_mask": [1] * len(ids1),
             "offsets": off1, "labels": gold_to_token_labels(["O"] * 2, off1),
             "dialogue_id": "d1", "sentence_id": "1", "text": "你好"},
            {"input_ids": ids2, "attention_mask": [1] * len(ids2),
             "offsets": off2, "labels": gold_to_token_labels(["O"] * 12, off2),
             "dialogue_id": "d2", "sentence_id": "2", "text": "你好，请问孩子发烧几天了"},
        ]
        out = collate_fn(batch)
        self.assertEqual(out["input_ids"].shape[1], len(ids2))
        self.assertEqual(out["attention_mask"][0].sum().item(), len(ids1))
        # 短句补位部分必须是 -100，否则会被算进损失
        pad = out["labels"][0][len(ids1):]
        self.assertTrue((pad == IGNORE_INDEX).all())


if __name__ == "__main__":
    unittest.main()
