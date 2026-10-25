"""Hand-calculated artificial tests for the train-only data profile.

No real train/dev/test examples or model predictions are used in these tests.
"""

import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
import profile_data as profile


def artificial_data():
    return {"synthetic_d": {"dialogue": [
        {"sentence_id": "01", "sentence": "甲乙丙丁戊己",
         "BIO_label": "B-Symptom I-Symptom B-Symptom O B-Drug I-Drug"},
        {"sentence_id": "02", "sentence": "甲乙", "BIO_label": "B-Symptom I-Symptom"},
        {"sentence_id": "03", "sentence": "。 ", "BIO_label": "O O"},
        {"sentence_id": "04", "sentence": "A 😀，𠀀é",
         "BIO_label": "O O B-Operation O B-Drug I-Drug O"},
        {"sentence_id": "05", "sentence": "", "BIO_label": ""},
    ]}}


def artificial_split():
    data = {
        "train_a": {"dialogue": [{"sentence_id": "1", "sentence": "甲乙", "BIO_label": "B-Symptom I-Symptom"}]},
        "train_b": {"dialogue": [{"sentence_id": "1", "sentence": "丙", "BIO_label": "O"}]},
        # No BIO at all: split integrity checks must not require/analyze val labels.
        "val_a": {"dialogue": [{"sentence_id": "1", "sentence": "丁戊"}]},
    }
    manifest = {
        "source_file": "IMCS-V2_train.json", "source_sha256": "a" * 64,
        "n_dialogues": 3, "train": ["train_a", "train_b"], "val": ["val_a"],
        "stats": {"train": {"dialogues": 2, "sentences": 2, "chars": 3},
                  "val": {"dialogues": 1, "sentences": 1, "chars": 2}},
    }
    return data, manifest


class ProfileDataTests(unittest.TestCase):
    def test_occurrences_differ_from_sentence_support(self):
        stats = profile.analyze(artificial_data())
        rows = {r["entity_type"]: r for r in stats["entity_types"]}
        self.assertEqual(stats["totals"]["entities"], 6)
        self.assertEqual(rows["Symptom"]["entities"], 3)
        self.assertEqual(rows["Symptom"]["sentences_containing_type"], 2)
        self.assertEqual(rows["Drug"]["entities"], 2)
        self.assertEqual(rows["Operation"]["entities"], 1)
        self.assertAlmostEqual(rows["Symptom"]["entity_share"], 1 / 2, delta=1e-12)
        self.assertEqual(stats["entity_length_summary"]["count"], 6)
        self.assertAlmostEqual(stats["entity_length_summary"]["mean"], 10 / 6, delta=1e-12)
        self.assertEqual(stats["entity_length_summary"]["median"], 2)
        self.assertEqual(stats["totals"]["entity_chars"], 10)
        self.assertAlmostEqual(stats["totals"]["o_character_share"], 7 / 17, delta=1e-12)
        # Two categories are absent: an imbalance ratio with denominator 0 is undefined.
        self.assertIsNone(stats["totals"]["largest_smallest_type_ratio"])
        self.assertTrue(all(stats["checks"].values()))

    def test_adjacency_multitype_and_entity_count_histogram(self):
        stats = profile.analyze(artificial_data())
        self.assertEqual(stats["totals"]["adjacent_pairs"], 1)
        self.assertEqual(stats["totals"]["adjacent_same_type_pairs"], 1)
        self.assertEqual(stats["adjacent_type_pairs"], [
            {"left_type": "Symptom", "right_type": "Symptom", "pairs": 1}])
        self.assertEqual(stats["totals"]["multi_entity"], 2)
        self.assertEqual(stats["totals"]["multi_type"], 2)
        self.assertEqual(stats["totals"]["no_entity"], 2)
        self.assertEqual({r["entities"]: r["sentences"] for r in stats["entities_per_sentence"]},
                         {0: 2, 1: 1, 2: 1, 3: 1})
        first_bin = stats["sentence_length_bins"][0]
        self.assertEqual(first_bin["sentences"], 5)
        self.assertAlmostEqual(first_bin["multi_entity_share"], .4, delta=1e-12)
        self.assertAlmostEqual(first_bin["entities_per_100_chars"], 600 / 17, delta=1e-12)

    def test_unicode_punctuation_spaces_and_empty_sentence(self):
        data = artificial_data()
        self.assertEqual(len(data["synthetic_d"]["dialogue"][3]["sentence"]), 7)
        stats = profile.analyze(data)
        self.assertEqual(stats["totals"]["sentences"], 5)
        self.assertEqual(stats["totals"]["chars"], 17)
        self.assertEqual(stats["totals"]["empty"], 1)
        self.assertEqual(stats["sentence_length_summary"]["median"], 2)
        self.assertEqual(stats["sentence_length_summary"]["max"], 7)

    def test_empty_data_and_no_entity_distribution_are_explicit(self):
        stats = profile.analyze({})
        self.assertEqual(stats["totals"]["sentences"], 0)
        self.assertEqual(stats["totals"]["entities"], 0)
        self.assertIsNone(stats["entity_length_summary"]["median"])
        self.assertIsNone(stats["sentence_length_summary"]["p95"])
        self.assertEqual(stats["totals"]["no_entity_share"], 0.0)
        self.assertIsNone(stats["totals"]["largest_smallest_type_ratio"])
        self.assertTrue(all(stats["checks"].values()))

    def test_invalid_bio_and_mismatched_length_are_rejected(self):
        for text, bio in [("甲", "I-Symptom"), ("甲乙", "B-Symptom I-Drug"),
                          ("甲", "B-Unknown"), ("甲乙", "O"), ("甲乙", "O  O")]:
            with self.subTest(bio=bio):
                data = {"synthetic": {"dialogue": [{"sentence_id": "1", "sentence": text, "BIO_label": bio}]}}
                with self.assertRaises(ValueError):
                    profile.analyze(data)

    def test_quantile_has_defined_interpolation_and_empty_policy(self):
        self.assertAlmostEqual(profile.quantile([4, 1, 3, 2], .9), 3.7, delta=1e-12)
        self.assertEqual(profile.quantile([4, 1, 3, 2], 0), 1)
        self.assertEqual(profile.quantile([4, 1, 3, 2], 1), 4)
        self.assertEqual(profile.quantile([8], .95), 8)
        self.assertIsNone(profile.quantile([], .95))
        with self.assertRaises(ValueError):
            profile.quantile([1], 1.1)

    def test_fixed_bins_cover_inclusive_edges(self):
        entity_expected = {1: "1", 2: "2", 3: "3–4", 4: "3–4", 5: "5–8", 8: "5–8", 9: "9+", 100: "9+"}
        for value, expected in entity_expected.items():
            self.assertEqual(profile.bin_name(value, profile.ENTITY_BINS), expected)
        for value, expected in {0: "0–10", 10: "0–10", 11: "11–20", 20: "11–20", 21: "21–40",
                                40: "21–40", 41: "41–80", 80: "41–80", 81: "81–160", 160: "81–160", 161: "161+"}.items():
            self.assertEqual(profile.bin_name(value, profile.SENTENCE_BINS), expected)
        with self.assertRaises(ValueError):
            profile.bin_name(0, profile.ENTITY_BINS)

    def test_split_integrity_does_not_require_val_bio(self):
        data, manifest = artificial_split()
        result = profile.check_split(data, manifest, "a" * 64)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["disjoint"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["counts"]["train"], {"dialogues": 2, "sentences": 2, "chars": 3})
        self.assertEqual(result["counts"]["val"], {"dialogues": 1, "sentences": 1, "chars": 2})

    def test_duplicate_split_ids_and_json_keys_are_rejected(self):
        data, manifest = artificial_split()
        manifest["train"].append("train_a")
        with self.assertRaises(ValueError):
            profile.check_split(data, manifest, "a" * 64)
        with tempfile.TemporaryDirectory(dir=STUDY) as folder:
            path = Path(folder) / "split.json"
            path.write_text('{"train": [], "train": ["x"]}', encoding="utf-8")
            with self.assertRaises(ValueError):
                profile.read_split(path)

    def test_split_overlap_and_missing_dialogues_are_rejected(self):
        data, original = artificial_split()
        for val in [["train_a"], ["unknown"], []]:
            manifest = copy.deepcopy(original)
            manifest["val"] = val
            with self.subTest(val=val), self.assertRaises(ValueError):
                profile.check_split(data, manifest, "a" * 64)

    def test_split_source_and_declared_counts_must_match(self):
        data, original = artificial_split()
        wrong_source = copy.deepcopy(original)
        wrong_source["source_sha256"] = "b" * 64
        wrong_count = copy.deepcopy(original)
        wrong_count["stats"]["train"]["chars"] = 4
        wrong_total = copy.deepcopy(original)
        wrong_total["n_dialogues"] = 4
        for manifest in [wrong_source, wrong_count, wrong_total]:
            with self.assertRaises(ValueError):
                profile.check_split(data, manifest, "a" * 64)

    def test_csv_roundtrip_totals_and_no_overwrite(self):
        stats = profile.analyze(artificial_data())
        with tempfile.TemporaryDirectory(dir=STUDY) as folder:
            directory = Path(folder)
            profile.write_tables(stats, directory)
            with (directory / "entity_types.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(sum(int(r["entities"]) for r in rows), 6)
            self.assertEqual(len(rows), 5)
            with (directory / "sentence_complexity.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(sum(int(r["sentences"]) for r in csv.DictReader(stream)), 5)
            snapshot = (directory / "entity_types.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                profile.write_tables(stats, directory)
            self.assertEqual((directory / "entity_types.csv").read_bytes(), snapshot)
            self.assertEqual(len(list(directory.glob("*.csv"))), 5)


if __name__ == "__main__":
    unittest.main()
