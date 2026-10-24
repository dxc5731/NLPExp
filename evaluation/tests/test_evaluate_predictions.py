"""Hand-calculated artificial tests, never model experiments. Float delta=1e-12."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from src.evaluate_predictions import evaluate, save_result
from src.ner_metrics import (Entity, LABEL_ORDER, extract_entities, prf_counts,
                             score_confusion_matrix, score_entity_sets, score_sequences)
from src.validate_predictions import load_and_validate


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "scorer_validation"
TOL = 1e-12


def tags(value):
    return value.split(" ") if value else []


def one(gold, pred):
    return score_sequences({("d", "s"): tags(gold)}, {("d", "s"): tags(pred)})


class MetricAssertions(unittest.TestCase):
    def check_metric(self, metric, tp, fp, fn, precision, recall, f1):
        self.assertEqual((tp, fp, fn), (metric["tp"], metric["fp"], metric["fn"]))
        for key, expected in (("precision", precision), ("recall", recall), ("f1", f1)):
            self.assertAlmostEqual(expected, metric[key], delta=TOL, msg=key)
        self.assertEqual(tp + fn, metric["support"])
        self.assertEqual(tp + fp, metric["predicted"])


class EntityTests(MetricAssertions):
    def test_exact_match_all_five_types(self):
        seq = "B-Symptom I-Symptom O B-Drug I-Drug O B-Drug_Category O B-Medical_Examination O B-Operation"
        result = one(seq, seq)["entities"]
        self.check_metric(result["micro"], 5, 0, 0, 1, 1, 1)
        self.assertEqual((5, 5), (result["gold_total"], result["predicted_total"]))
        for metric in result["per_type"].values():
            self.check_metric(metric, 1, 0, 0, 1, 1, 1)

    def test_gold_entity_prediction_all_o(self):
        self.check_metric(one("B-Symptom I-Symptom", "O O")["entities"]["micro"], 0, 0, 1, 0, 0, 0)

    def test_no_gold_but_predicted_entity(self):
        self.check_metric(one("O O", "B-Drug I-Drug")["entities"]["micro"], 0, 1, 0, 0, 0, 0)

    def test_missing_one_of_two_entities(self):
        self.check_metric(one("B-Symptom O B-Drug", "B-Symptom O O")["entities"]["micro"],
                          1, 0, 1, 1, 1 / 2, 2 / 3)

    def test_wrong_type_counts_fp_and_fn(self):
        result = one("B-Symptom I-Symptom", "B-Drug I-Drug")["entities"]
        self.check_metric(result["micro"], 0, 1, 1, 0, 0, 0)
        self.check_metric(result["per_type"]["Symptom"], 0, 0, 1, 0, 0, 0)
        self.check_metric(result["per_type"]["Drug"], 0, 1, 0, 0, 0, 0)

    def test_wrong_start(self):
        self.check_metric(one("O B-Drug I-Drug", "B-Drug I-Drug I-Drug")["entities"]["micro"],
                          0, 1, 1, 0, 0, 0)

    def test_wrong_end(self):
        self.check_metric(one("B-Drug I-Drug O", "B-Drug I-Drug I-Drug")["entities"]["micro"],
                          0, 1, 1, 0, 0, 0)

    def test_half_open_spans_and_sentence_end(self):
        entities = extract_entities(tags("O B-Drug I-Drug O B-Operation"), "d", "07")
        self.assertEqual({Entity("d", "07", "Drug", 1, 3), Entity("d", "07", "Operation", 4, 5)}, entities)

    def test_adjacent_same_type_remains_two_entities(self):
        seq = tags("B-Symptom B-Symptom I-Symptom")
        self.assertEqual({Entity("d", "s", "Symptom", 0, 1), Entity("d", "s", "Symptom", 1, 3)},
                         extract_entities(seq, "d", "s"))
        self.check_metric(one("B-Symptom B-Symptom", "B-Symptom I-Symptom")["entities"]["micro"],
                          0, 1, 2, 0, 0, 0)

    def test_same_span_cannot_match_across_sentences(self):
        g = {("d", "1"): ["B-Drug"], ("d", "2"): ["O"]}
        p = {("d", "1"): ["O"], ("d", "2"): ["B-Drug"]}
        self.check_metric(score_sequences(g, p)["entities"]["micro"], 0, 1, 1, 0, 0, 0)

    def test_same_span_cannot_match_across_dialogues(self):
        g = {("a", "1"): ["B-Drug"], ("b", "1"): ["O"]}
        p = {("a", "1"): ["O"], ("b", "1"): ["B-Drug"]}
        self.check_metric(score_sequences(g, p)["entities"]["micro"], 0, 1, 1, 0, 0, 0)

    def test_tuple_ids_avoid_underscore_collision(self):
        g = {("a_b", "c"): ["B-Drug"], ("a", "b_c"): ["O"]}
        p = {("a_b", "c"): ["O"], ("a", "b_c"): ["B-Drug"]}
        self.check_metric(score_sequences(g, p)["entities"]["micro"], 0, 1, 1, 0, 0, 0)

    def test_mixed_five_types_manual_expected(self):
        result = evaluate(FIXTURE / "data_dev.json", FIXTURE / "predictions_dev.json", purpose="scorer_validation")
        self.assertEqual("SCORED", result["status"])
        self.check_metric(result["entities"]["micro"], 2, 2, 3, 1 / 2, 2 / 5, 4 / 9)
        expected = {"Symptom": (1, 0, 0, 1, 1, 1), "Drug": (0, 1, 1, 0, 0, 0),
                    "Drug_Category": (0, 1, 1, 0, 0, 0), "Medical_Examination": (0, 0, 1, 0, 0, 0),
                    "Operation": (1, 0, 0, 1, 1, 1)}
        for kind, numbers in expected.items():
            self.check_metric(result["entities"]["per_type"][kind], *numbers)
        self.assertEqual((5, 4), (result["entities"]["gold_total"], result["entities"]["predicted_total"]))

    def test_both_all_o_zero_denominators(self):
        result = one("O O", "O O")["entities"]
        self.check_metric(result["micro"], 0, 0, 0, 0, 0, 0)
        for metric in result["per_type"].values():
            self.check_metric(metric, 0, 0, 0, 0, 0, 0)

    def test_empty_pure_inputs_no_nan(self):
        result = score_sequences({}, {})
        self.check_metric(result["entities"]["micro"], 0, 0, 0, 0, 0, 0)
        self.assertEqual(0, result["characters"]["accuracy"])
        self.assertEqual(0, result["characters"]["including_o"]["macro_f1"])
        json.dumps(result, allow_nan=False)

    def test_global_micro_is_not_sentence_mean(self):
        g = {("d", "1"): tags("B-Drug"), ("d", "2"): tags("B-Drug B-Drug B-Drug")}
        p = {("d", "1"): tags("O"), ("d", "2"): tags("B-Drug B-Drug B-Drug")}
        self.check_metric(score_sequences(g, p)["entities"]["micro"], 3, 0, 1, 1, 3 / 4, 6 / 7)
        self.assertNotAlmostEqual(1 / 2, score_sequences(g, p)["entities"]["micro"]["f1"], delta=TOL)

    def test_illegal_bio_rejected_by_pure_api(self):
        for seq in ("I-Drug", "O I-Drug", "B-Symptom I-Drug", "PAD"):
            with self.subTest(seq=seq), self.assertRaises(ValueError):
                extract_entities(tags(seq), "d", "s")
        with self.assertRaises(ValueError):
            extract_entities("O O", "d", "s")

    def test_pure_api_refuses_id_and_length_mismatch(self):
        with self.assertRaises(ValueError):
            score_sequences({("d", "1"): ["O"]}, {("d", "2"): ["O"]})
        with self.assertRaises(ValueError):
            score_sequences({("d", "1"): ["O"]}, {("d", "1"): []})


class CharacterTests(MetricAssertions):
    def test_perfect_labels_fixed_macro_denominator(self):
        result = one("B-Symptom I-Symptom O", "B-Symptom I-Symptom O")["characters"]
        self.assertEqual((3, 3, 1), (result["total"], result["correct"], result["accuracy"]))
        self.check_metric(result["including_o"]["micro"], 3, 0, 0, 1, 1, 1)
        self.assertAlmostEqual(3 / 11, result["including_o"]["macro_f1"], delta=TOL)
        self.assertAlmostEqual(2 / 10, result["excluding_o"]["macro_f1"], delta=TOL)
        self.check_metric(result["per_label"]["B-Symptom"], 1, 0, 0, 1, 1, 1)
        self.check_metric(result["per_label"]["B-Drug"], 0, 0, 0, 0, 0, 0)

    def test_all_o_with_gold_entities(self):
        result = one("B-Symptom I-Symptom O O", "O O O O")["characters"]
        self.assertEqual(1 / 2, result["accuracy"])
        self.check_metric(result["per_label"]["O"], 2, 2, 0, 1 / 2, 1, 2 / 3)
        self.check_metric(result["excluding_o"]["micro"], 0, 0, 2, 0, 0, 0)
        self.assertAlmostEqual(2 / 33, result["including_o"]["macro_f1"], delta=TOL)

    def test_only_b_i_boundary_error(self):
        result = one("B-Symptom I-Symptom", "B-Symptom B-Symptom")
        self.assertEqual(1 / 2, result["characters"]["accuracy"])
        self.check_metric(result["characters"]["per_label"]["B-Symptom"], 1, 1, 0, 1 / 2, 1, 2 / 3)
        self.check_metric(result["characters"]["per_label"]["I-Symptom"], 0, 0, 1, 0, 0, 0)
        self.check_metric(result["entities"]["micro"], 0, 2, 1, 0, 0, 0)

    def test_wrong_type_character_statistics(self):
        result = one("B-Symptom I-Symptom O", "B-Drug I-Drug O")["characters"]
        self.assertAlmostEqual(1 / 3, result["accuracy"], delta=TOL)
        self.check_metric(result["including_o"]["micro"], 1, 2, 2, 1 / 3, 1 / 3, 1 / 3)
        self.check_metric(result["excluding_o"]["micro"], 0, 2, 2, 0, 0, 0)
        self.assertAlmostEqual(1 / 11, result["including_o"]["macro_f1"], delta=TOL)

    def test_many_o_high_accuracy_zero_entity_f1(self):
        result = one(" ".join(["O"] * 99 + ["B-Drug"]), " ".join(["O"] * 100))
        self.assertEqual(0.99, result["characters"]["accuracy"])
        self.assertEqual(0.0, result["entities"]["micro"]["f1"])
        self.check_metric(result["characters"]["excluding_o"]["micro"], 0, 0, 1, 0, 0, 0)

    def test_no_o_summary_preserves_false_positives_and_false_negatives(self):
        result = one("B-Symptom I-Symptom O O", "B-Symptom O B-Drug O")["characters"]
        self.check_metric(result["excluding_o"]["micro"], 1, 1, 1, 1 / 2, 1 / 2, 1 / 2)
        self.assertAlmostEqual(1 / 10, result["excluding_o"]["macro_f1"], delta=TOL)
        self.assertAlmostEqual(3 / 22, result["including_o"]["macro_f1"], delta=TOL)
        self.check_metric(result["per_label"]["I-Symptom"], 0, 0, 1, 0, 0, 0)
        self.check_metric(result["per_label"]["B-Drug"], 0, 1, 0, 0, 0, 0)

    def test_confusion_direction_fixed_order_and_counts(self):
        result = one("B-Symptom I-Symptom O O", "B-Symptom O B-Drug O")["characters"]
        expected_order = ("O", "B-Symptom", "I-Symptom", "B-Drug", "I-Drug", "B-Drug_Category",
                          "I-Drug_Category", "B-Medical_Examination", "I-Medical_Examination", "B-Operation", "I-Operation")
        self.assertEqual(expected_order, LABEL_ORDER)
        cm = result["confusion_matrix"]
        expected = [[0] * 11 for _ in range(11)]
        expected[0][0] = expected[0][3] = expected[1][1] = expected[2][0] = 1
        self.assertEqual(expected, cm["counts"])
        self.assertEqual([2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], cm["row_sums"])
        self.assertEqual([2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], cm["column_sums"])
        self.assertEqual(("gold", "predicted"), (cm["rows"], cm["columns"]))
        self.assertEqual(4, sum(m["support"] for m in result["per_label"].values()))

    def test_mixed_fixture_manual_character_expectations(self):
        r = evaluate(FIXTURE / "data_dev.json", FIXTURE / "predictions_dev.json", purpose="scorer_validation")["characters"]
        self.assertEqual((22, 17), (r["total"], r["correct"]))
        self.assertAlmostEqual(17 / 22, r["accuracy"], delta=TOL)
        self.check_metric(r["including_o"]["micro"], 17, 5, 5, 17 / 22, 17 / 22, 17 / 22)
        self.check_metric(r["excluding_o"]["micro"], 5, 2, 5, 5 / 7, 1 / 2, 10 / 17)
        self.check_metric(r["per_label"]["O"], 12, 3, 0, 4 / 5, 1, 8 / 9)
        self.assertAlmostEqual(53 / 99, r["including_o"]["macro_f1"], delta=TOL)
        self.assertAlmostEqual(1 / 2, r["excluding_o"]["macro_f1"], delta=TOL)

    def test_unicode_space_punctuation_positions(self):
        r = evaluate(FIXTURE / "data_dev.json", FIXTURE / "predictions_dev.json", purpose="scorer_validation")
        self.assertEqual(22, r["counts"]["characters"])
        self.assertEqual(7, len("A 😀，𠀀e\u0301"))
        unicode_only = one("O O O O O O O", "O O O O O O O")["characters"]
        self.assertEqual(7, unicode_only["per_label"]["O"]["support"])
        self.assertEqual(1, unicode_only["accuracy"])

    def test_empty_matrix_zero_policy_and_shape_validation(self):
        r = score_confusion_matrix([[0] * 11 for _ in range(11)])
        self.check_metric(r["including_o"]["micro"], 0, 0, 0, 0, 0, 0)
        self.assertEqual(0, r["accuracy"])
        for matrix in ([], [[0]], [[-1] * 11 for _ in range(11)], [[0.0] * 11 for _ in range(11)]):
            with self.subTest(matrix=matrix[:1]), self.assertRaises(ValueError):
                score_confusion_matrix(matrix)


class IntegrationTests(MetricAssertions):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.data_path = self.folder / "source.json"
        self.pred_path = self.folder / "predictions_dev.json"
        self.data = json.loads((FIXTURE / "data_dev.json").read_text(encoding="utf-8"))
        self.pred = json.loads((FIXTURE / "predictions_dev.json").read_text(encoding="utf-8"))
        self.write()

    def write(self):
        self.data_path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        self.pred_path.write_text(json.dumps(self.pred, ensure_ascii=False), encoding="utf-8")

    def run_eval(self, split="dev"):
        return evaluate(self.data_path, self.pred_path, split=split, purpose="scorer_validation")

    def assert_rejected(self, expected_code):
        with patch("src.evaluate_predictions.ner_metrics.score_sequences", side_effect=AssertionError("must not score")):
            result = self.run_eval()
        self.assertEqual("REJECTED", result["status"])
        self.assertEqual(expected_code, result["rejection"]["code"])
        for key in ("entities", "characters", "metric_policy"):
            self.assertNotIn(key, result)
        self.assertIsNone(result["timing_seconds"]["scoring"])
        return result

    def cli(self, *, split="dev", output=None):
        return subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "src/evaluate_predictions.py"),
                               "--data", str(self.data_path), "--predictions", str(self.pred_path),
                               "--split", split, "--purpose", "scorer_validation", "--output-dir",
                               str(output or self.folder / "reports")],
                              capture_output=True, encoding="utf-8", cwd=ROOT)

    def test_valid_dev_readonly_hashes_and_counts(self):
        before = [p.read_bytes() for p in (self.data_path, self.pred_path)]
        result = self.run_eval()
        self.assertEqual("SCORED", result["status"])
        self.assertEqual({"dialogues": 1, "sentences": 2, "characters": 22}, result["counts"])
        self.assertEqual(before, [p.read_bytes() for p in (self.data_path, self.pred_path)])
        for name, raw in zip(("data", "predictions"), before):
            self.assertEqual(hashlib.sha256(raw).hexdigest(), result["inputs"][name]["sha256"])
        self.check_metric(result["entities"]["micro"], 2, 2, 3, 1 / 2, 2 / 5, 4 / 9)
        self.assertEqual("scorer_validation", result["purpose"])
        self.assertEqual("entities.micro.f1", result["metric_policy"]["primary_metric"])

    def test_missing_sentence_stops_scoring(self):
        del self.pred["manual_scoring"]["09"]
        self.write()
        result = self.assert_rejected("VALIDATION_FAIL")
        self.assertEqual(1, result["validation"]["errors"]["by_code"]["MISSING_SENTENCE"])

    def test_wrong_id_stops_scoring(self):
        self.pred["manual_scoring"]["9"] = self.pred["manual_scoring"].pop("09")
        self.write()
        self.assert_rejected("VALIDATION_FAIL")

    def test_length_error_stops_scoring(self):
        self.pred["manual_scoring"]["09"] = "O"
        self.write()
        self.assert_rejected("VALIDATION_FAIL")

    def test_invalid_label_stops_scoring(self):
        self.pred["manual_scoring"]["09"] = "PAD O O O O O O"
        self.write()
        self.assert_rejected("VALIDATION_FAIL")

    def test_illegal_bio_stops_scoring(self):
        self.pred["manual_scoring"]["09"] = "I-Drug O O O O O O"
        self.write()
        self.assert_rejected("VALIDATION_FAIL")

    def test_duplicate_key_stops_scoring(self):
        self.pred_path.write_text('{"manual_scoring":{},"manual_scoring":{}}', encoding="utf-8")
        self.assert_rejected("VALIDATION_FAIL")

    def test_bad_gold_stops_scoring(self):
        self.data["manual_scoring"]["dialogue"][0]["BIO_label"] = "I-Drug"
        self.write()
        self.assert_rejected("VALIDATION_FAIL")

    def test_raw_test_no_gold_refuses_even_after_validation_pass(self):
        for row in self.data["manual_scoring"]["dialogue"]:
            del row["BIO_label"]
        self.write()
        with patch("src.evaluate_predictions.ner_metrics.score_sequences", side_effect=AssertionError("must not score")):
            result = self.run_eval(split="test")
        self.assertEqual("PASS", result["validation"]["status"])
        self.assertEqual("TEST_HAS_NO_PUBLIC_GOLD", result["rejection"]["code"])
        self.assertNotIn("entities", result)
        self.assertEqual(2, self.cli(split="test").returncode)

    def test_template_not_run_refuses_scoring(self):
        self.data = {did: {sid: "" for sid in rows} for did, rows in self.pred.items()}
        self.write()
        with patch("src.evaluate_predictions.ner_metrics.score_sequences", side_effect=AssertionError("must not score")):
            result = self.run_eval(split="test")
        self.assertEqual("NOT_RUN", result["validation"]["status"])
        self.assertEqual("VALIDATION_NOT_RUN", result["rejection"]["code"])
        self.assertNotIn("characters", result)

    def test_cli_success_matches_expected(self):
        process = self.cli()
        self.assertEqual(0, process.returncode, process.stderr)
        paths = list((self.folder / "reports").glob("*/evaluation.json"))
        self.assertEqual(1, len(paths))
        result = json.loads(paths[0].read_text(encoding="utf-8"))
        self.assertEqual("SCORED", result["status"])
        self.assertAlmostEqual(4 / 9, result["entities"]["micro"]["f1"], delta=TOL)
        self.assertIn("ENTITY MICRO (primary)", process.stdout)

    def test_cli_failure_publishes_only_rejection_no_metrics(self):
        self.pred_path.write_bytes(b"null")
        process = self.cli()
        self.assertEqual(1, process.returncode)
        root = self.folder / "reports"
        self.assertEqual([], list(root.glob("*/evaluation.json")))
        paths = list(root.glob("*/rejection.json"))
        self.assertEqual(1, len(paths))
        result = json.loads(paths[0].read_text(encoding="utf-8"))
        self.assertEqual("REJECTED", result["status"])
        self.assertNotIn("entities", result)
        self.assertNotIn("characters", result)

    def test_repeat_publication_never_overwrites(self):
        result = self.run_eval()
        a = save_result(result, self.folder / "out", data_path=self.data_path, prediction_path=self.pred_path)
        before = a.read_bytes()
        b = save_result(result, self.folder / "out", data_path=self.data_path, prediction_path=self.pred_path)
        self.assertNotEqual(a, b)
        self.assertEqual(before, a.read_bytes())
        self.assertEqual(before, b.read_bytes())

    def test_output_input_protection(self):
        before = self.pred_path.read_bytes()
        result = self.run_eval()
        with self.assertRaises(ValueError):
            save_result(result, self.pred_path, data_path=self.data_path, prediction_path=self.pred_path)
        with self.assertRaises(ValueError):
            save_result(result, self.folder / "out", data_path=self.folder, prediction_path=self.pred_path)
        self.assertEqual(before, self.pred_path.read_bytes())

    def test_zip_and_directory_reuse(self):
        data_dir = self.folder / "data"
        data_dir.mkdir()
        (data_dir / "IMCS-V2_dev.json").write_bytes(self.data_path.read_bytes())
        archive = self.folder / "data.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("IMCS-V2-NER/IMCS-V2_dev.json", self.data_path.read_bytes())
        before = archive.read_bytes()
        for source in (archive, data_dir):
            with self.subTest(source=source):
                result = evaluate(source, self.pred_path, purpose="scorer_validation")
                self.assertAlmostEqual(4 / 9, result["entities"]["micro"]["f1"], delta=TOL)
        self.assertEqual(before, archive.read_bytes())

    def test_single_loaded_snapshot_no_reread_after_validation(self):
        real_loader = load_and_validate

        def loader(*args, **kwargs):
            snapshot = real_loader(*args, **kwargs)
            self.pred_path.write_bytes(b"null")
            return snapshot

        original = self.pred_path.read_bytes()
        with patch("src.evaluate_predictions.validate_predictions.load_and_validate", side_effect=loader):
            result = self.run_eval()
        self.assertEqual("SCORED", result["status"])
        self.assertEqual(hashlib.sha256(original).hexdigest(), result["inputs"]["predictions"]["sha256"])
        self.assertAlmostEqual(4 / 9, result["entities"]["micro"]["f1"], delta=TOL)




if __name__ == "__main__":
    unittest.main()
