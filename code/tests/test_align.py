# -*- coding: utf-8 -*-
"""对齐层的自动测试（v2）。

跑法（在 code/ 目录下）：
    python -m unittest discover -s tests -v

这些测试回答的是「对齐会不会静默出错」：
错位不报错、只掉分，所以必须用断言钉死；而 v1 的两个静默动作
（I 自动改 B、没覆盖的字符自动补 O）现在都必须是显式报错或显式计数。
"""

import unittest

from src.align import (
    IGNORE_INDEX,
    AlignmentError,
    encode_sentence,
    load_tokenizer,
    restore_char_labels,
    to_legal_bio,
)
from src.data_io import load_split
from src.dataset import collate_fn
from src.labels import LABEL2ID


class AlignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = load_tokenizer()

    def test_offsets_cover_every_char_once(self):
        text = "孩子发烧38.5度，吃了小儿吗丁啉"
        enc = encode_sentence(self.tok, text)
        covered = [0] * len(text)
        for s, e in enc.offsets:
            for k in range(s, e):
                covered[k] += 1
        self.assertEqual(covered, [1] * len(text))

    def test_special_tokens_get_ignore_index(self):
        text = "您好"
        enc = encode_sentence(self.tok, text, tags=["O"] * len(text))
        self.assertEqual(enc.token_labels[0], IGNORE_INDEX)
        self.assertEqual(enc.token_labels[-1], IGNORE_INDEX)
        self.assertEqual(len(enc.token_labels), len(enc.input_ids))

    def test_representable_multichar_token_keeps_label(self):
        """'37' 覆盖 B-Symptom + I-Symptom：一个标签摊回去能逐字还原，不算歧义。"""
        text = "37"
        enc = encode_sentence(self.tok, text, tags=["B-Symptom", "I-Symptom"])
        self.assertEqual(enc.ambiguous, [])
        real = [i for i, (s, e) in enumerate(enc.offsets) if s != e and e - s > 1]
        self.assertEqual(enc.token_labels[real[0]], LABEL2ID["B-Symptom"])

    def test_non_representable_token_is_marked_ambiguous(self):
        """'838' 覆盖 I + B + I：任何一个标签都还原不出中间的 B，必须标 -100。"""
        text = "838"
        tags = ["I-Symptom", "B-Symptom", "I-Symptom"]
        enc = encode_sentence(self.tok, text, tags=tags)
        self.assertEqual(len(enc.ambiguous), 1)
        self.assertEqual(enc.ambiguous[0]["text"], "838")
        self.assertEqual(enc.ambiguous[0]["tags"], tags)
        real = [i for i, (s, e) in enumerate(enc.offsets) if s != e]
        self.assertTrue(all(enc.token_labels[i] == IGNORE_INDEX for i in real))

    def test_o_then_b_inside_token_is_ambiguous(self):
        text = "363"
        enc = encode_sentence(self.tok, text, tags=["O", "O", "B-Symptom"])
        self.assertEqual(len(enc.ambiguous), 1)

    def test_restore_does_not_repair_illegal_bio(self):
        """句首的 I-X 必须原样返回，不能自动改成 B-X。"""
        restored = restore_char_labels([LABEL2ID["I-Symptom"]], [(0, 2)], 2)
        self.assertEqual(restored, ["I-Symptom", "I-Symptom"])

    def test_restore_returns_none_for_uncovered_chars(self):
        """没被覆盖的字符必须是 None，不能悄悄补 O。"""
        restored = restore_char_labels([LABEL2ID["O"]], [(0, 2)], 3)
        self.assertIsNone(restored[2])

    def test_to_legal_bio_counts_changes(self):
        fixed, n = to_legal_bio(["O", "I-Symptom", "I-Symptom", "O", "I-Drug"])
        self.assertEqual(fixed, ["O", "B-Symptom", "I-Symptom", "O", "B-Drug"])
        self.assertEqual(n, 2)

    def test_to_legal_bio_rejects_none(self):
        with self.assertRaises(AlignmentError):
            to_legal_bio(["O", None])

    def test_long_sentence_raises_instead_of_truncating(self):
        text = "咳" * 400
        with self.assertRaises(AlignmentError):
            encode_sentence(self.tok, text, max_len=256)

    def test_roundtrip_on_plain_sentence(self):
        text = "但不能祛痰，所以建议加上氨溴索嘛"
        tags = ("O O O O B-Symptom O O O O O O O "
                "B-Drug I-Drug I-Drug O").split(" ")
        enc = encode_sentence(self.tok, text, tags=tags)
        self.assertEqual(enc.ambiguous, [])
        restored = restore_char_labels(enc.token_labels, enc.offsets, len(text))
        self.assertEqual(restored, tags)

    def test_known_ambiguous_sentences_in_train(self):
        """回归：训练集里这 8 句各有一个无法无损映射的词片，数量不能变。"""
        known = [
            ("10034734", "2"), ("10098865", "21"), ("10193361", "25"),
            ("10206772", "25"), ("10491432", "7"), ("10506585", "10"),
            ("10515756", "10"), ("10731191", "31"),
        ]
        data = load_split("train")
        for did, sid in known:
            sent = next(s for s in data[did]["dialogue"] if s["sentence_id"] == sid)
            enc = encode_sentence(self.tok, sent["sentence"],
                                  tags=sent["BIO_label"].split(" "),
                                  where=f"对话{did}/句{sid}")
            self.assertGreaterEqual(len(enc.ambiguous), 1, f"{did}/{sid} 应当有歧义词片")

    def test_collate_pads_with_ignore_index(self):
        e1 = encode_sentence(self.tok, "你好", tags=["O", "O"])
        e2 = encode_sentence(self.tok, "你好，请问孩子发烧几天了", tags=["O"] * 12)
        batch = [
            {"input_ids": e1.input_ids, "attention_mask": [1] * len(e1.input_ids),
             "offsets": e1.offsets, "labels": e1.token_labels,
             "dialogue_id": "d1", "sentence_id": "1", "text": "你好"},
            {"input_ids": e2.input_ids, "attention_mask": [1] * len(e2.input_ids),
             "offsets": e2.offsets, "labels": e2.token_labels,
             "dialogue_id": "d2", "sentence_id": "2", "text": "你好，请问孩子发烧几天了"},
        ]
        out = collate_fn(batch)
        self.assertEqual(out["input_ids"].shape[1], len(e2.input_ids))
        self.assertEqual(out["attention_mask"][0].sum().item(), len(e1.input_ids))
        pad = out["labels"][0][len(e1.input_ids):]
        self.assertTrue((pad == IGNORE_INDEX).all())


if __name__ == "__main__":
    unittest.main()
