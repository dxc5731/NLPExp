"""Synthetic contract tests only: no models, official predictions, or scores."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from src.validate_predictions import ENTITY_TYPES, LABELS, run_validation


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "validate_predictions.py"


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        # Noncontiguous, leading-zero, and underscore IDs must remain literal.
        self.data = {
            "d_01": {"dialogue": [
                {"sentence_id": "01", "sentence": "甲，", "BIO_label": "B-Symptom O"},
                {"sentence_id": "07", "sentence": "药乙", "BIO_label": "B-Drug I-Drug"},
            ]},
            "d_02": {"dialogue": [
                {"sentence_id": "9", "sentence": "好", "BIO_label": "O"},
            ]},
        }
        self.pred = {did: {r["sentence_id"]: r["BIO_label"] for r in d["dialogue"]}
                     for did, d in self.data.items()}
        self.data_path = self.folder / "source.json"
        self.pred_path = self.folder / "predictions_dev.json"

    def write_json(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def validate(self, *, data=None, pred=None, split="dev", max_examples=30):
        self.write_json(self.data_path, self.data if data is None else data)
        self.write_json(self.pred_path, self.pred if pred is None else pred)
        return run_validation(self.data_path, self.pred_path, split=split, max_examples=max_examples)

    def assert_error(self, result, code):
        self.assertEqual("FAIL", result["status"], result)
        self.assertGreater(result["errors"]["by_code"].get(code, 0), 0, result)

    def assert_bad_sequence(self, value, code):
        self.pred["d_01"]["07"] = value
        self.assert_error(self.validate(), code)

    def cli(self, *extra):
        return subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT), *map(str, extra)],
                              capture_output=True, encoding="utf-8", cwd=ROOT)

    def test_legal_complete_predictions(self):
        result = self.validate()
        self.assertEqual("PASS", result["status"])
        self.assertEqual({"passed": 3, "failed": 0, "not_fully_checked": 0},
                         result["predictions"]["expected_sentences"])
        self.assertEqual((2, 3), (result["data"]["dialogues"], result["data"]["sentences"]))
        self.assertEqual(11, len(LABELS))

    def test_all_five_types_and_adjacent_entities(self):
        tags = [tag for t in ENTITY_TYPES for tag in ("B-" + t, "I-" + t)]
        tags += ["B-Drug", "B-Drug", "I-Drug", "I-Drug", "O"]
        sequence = " ".join(tags)
        data = {"1": {"dialogue": [{"sentence_id": "8", "sentence": "甲" * len(tags),
                                     "BIO_label": sequence}]}}
        self.assertEqual("PASS", self.validate(data=data, pred={"1": {"8": sequence}})["status"])

    def test_missing_sentence(self):
        del self.pred["d_01"]["07"]
        result = self.validate()
        self.assert_error(result, "MISSING_SENTENCE")
        self.assertEqual(1, result["predictions"]["expected_sentences"]["failed"])

    def test_extra_sentence(self):
        self.pred["d_01"]["extra"] = "O"
        self.assert_error(self.validate(), "EXTRA_SENTENCE")

    def test_wrong_id_with_unchanged_count(self):
        self.pred["d_01"]["1"] = self.pred["d_01"].pop("01")
        result = self.validate()
        self.assert_error(result, "MISSING_SENTENCE")
        self.assert_error(result, "EXTRA_SENTENCE")
        self.assertEqual("PASS", result["checks"]["record_counts"]["status"])

    def test_missing_dialogue(self):
        del self.pred["d_01"]
        self.assert_error(self.validate(), "MISSING_DIALOGUE")

    def test_extra_empty_dialogue(self):
        self.pred["extra"] = {}
        self.assert_error(self.validate(), "EXTRA_DIALOGUE")

    def test_wrong_dialogue_id(self):
        self.pred["other"] = self.pred.pop("d_01")
        result = self.validate()
        self.assert_error(result, "MISSING_DIALOGUE")
        self.assert_error(result, "EXTRA_DIALOGUE")

    def test_empty_predictions(self):
        self.assert_error(self.validate(pred={}), "MISSING_SENTENCE")

    def test_short_and_long_sequences(self):
        for seq in ("O", "O O O", ""):
            with self.subTest(seq=seq):
                self.assert_bad_sequence(seq, "LABEL_LENGTH_MISMATCH")

    def test_no_mutation_or_silent_repair(self):
        self.write_json(self.data_path, self.data)
        self.pred["d_01"]["07"] = "O"
        self.write_json(self.pred_path, self.pred)
        before = [p.read_bytes() for p in (self.data_path, self.pred_path)]
        result = run_validation(self.data_path, self.pred_path, split="dev")
        self.assert_error(result, "LABEL_LENGTH_MISMATCH")
        self.assertEqual(before, [p.read_bytes() for p in (self.data_path, self.pred_path)])
        self.assertEqual(hashlib.sha256(before[1]).hexdigest(), result["inputs"]["predictions"]["sha256"])

    def test_unknown_and_special_labels(self):
        for tag in ("B-Disease", "PAD", "[CLS]", "[SEP]", "[PAD]", "o", "B-symptom"):
            with self.subTest(tag=tag):
                self.assert_bad_sequence(tag + " O", "INVALID_LABEL")

    def test_i_at_sentence_start(self):
        self.assert_bad_sequence("I-Drug I-Drug", "ILLEGAL_I_TRANSITION")

    def test_o_to_i(self):
        self.assert_bad_sequence("O I-Drug", "ILLEGAL_I_TRANSITION")

    def test_cross_type_i(self):
        self.assert_bad_sequence("B-Symptom I-Drug", "ILLEGAL_I_TRANSITION")

    def test_bio_resets_at_sentence_boundary(self):
        self.pred["d_01"]["01"] = "B-Drug I-Drug"
        self.pred["d_01"]["07"] = "I-Drug I-Drug"
        self.assert_error(self.validate(), "ILLEGAL_I_TRANSITION")

    def test_invalid_separators(self):
        for seq in (" O O", "O O ", "O  O", "O\tO", "O\nO", "O\u00a0O", "O\u3000O"):
            with self.subTest(seq=repr(seq)):
                self.assert_bad_sequence(seq, "INVALID_SEPARATOR")

    def test_unicode_codepoints_spaces_punctuation(self):
        text = "A 😀，𠀀e\u0301"
        self.assertEqual(7, len(text))
        self.assertGreater(len(text.encode("utf-16-le")) // 2, len(text))
        sequence = " ".join(["O"] * 7)
        data = {"fixture": {"dialogue": [{"sentence_id": "3", "sentence": text, "BIO_label": sequence}]}}
        result = self.validate(data=data, pred={"fixture": {"3": sequence}})
        self.assertEqual("PASS", result["status"])
        self.assertEqual(7, result["data"]["characters"])
        self.assert_error(self.validate(data=data, pred={"fixture": {"3": "O O O O O O"}}),
                          "LABEL_LENGTH_MISMATCH")

    def test_prediction_value_types(self):
        for value in (None, [], {}, 1, True):
            with self.subTest(value=value):
                self.pred["d_01"]["07"] = value
                self.assert_error(self.validate(), "PREDICTION_VALUE_TYPE")

    def test_prediction_root_types_including_null(self):
        self.write_json(self.data_path, self.data)
        for value in (None, [], "O", 1):
            with self.subTest(value=value):
                self.write_json(self.pred_path, value)
                self.assert_error(run_validation(self.data_path, self.pred_path, split="dev"),
                                  "PREDICTION_ROOT_TYPE")

    def test_prediction_dialogue_type_and_empty_ids(self):
        for value in ({"d_01": []}, {"": {}}, {"d_01": {"": "O"}}):
            with self.subTest(value=value):
                result = self.validate(pred=value)
                self.assertEqual("FAIL", result["status"])
                self.assertEqual("FAIL", result["checks"]["prediction_structure"]["status"])

    def test_duplicate_prediction_keys_root_and_nested(self):
        self.write_json(self.data_path, self.data)
        for raw in ('{"d_01":{},"d_01":{}}', '{"d_01":{"01":"O O","01":"O O"}}'):
            with self.subTest(raw=raw):
                self.pred_path.write_text(raw, encoding="utf-8")
                result = run_validation(self.data_path, self.pred_path, split="dev")
                self.assert_error(result, "DUPLICATE_JSON_KEY")
                sample = next(e for e in result["errors"]["examples"] if e["code"] == "DUPLICATE_JSON_KEY")
                self.assertTrue(sample["json_pointer"].startswith("/d_01"))

    def test_duplicate_source_key(self):
        self.data_path.write_text('{"d":{},"d":{}}', encoding="utf-8")
        self.write_json(self.pred_path, self.pred)
        self.assert_error(run_validation(self.data_path, self.pred_path, split="dev"), "DUPLICATE_JSON_KEY")

    def test_duplicate_source_sentence_id(self):
        self.data["d_01"]["dialogue"].append(copy.deepcopy(self.data["d_01"]["dialogue"][0]))
        self.assert_error(self.validate(), "DUPLICATE_SENTENCE_ID")

    def test_invalid_json_utf8_bom_and_constants(self):
        self.write_json(self.data_path, self.data)
        for raw, code in ((b'{', "INVALID_JSON"), (b'\xff', "INVALID_UTF8"),
                          (b'\xef\xbb\xbf{}', "UTF8_BOM"), (b'{"x":NaN}', "INVALID_JSON"),
                          (b'{"x":Infinity}', "INVALID_JSON")):
            with self.subTest(raw=raw):
                self.pred_path.write_bytes(raw)
                self.assert_error(run_validation(self.data_path, self.pred_path, split="dev"), code)

    def test_unpaired_surrogates_rejected_and_report_encodable(self):
        self.write_json(self.data_path, self.data)
        self.pred_path.write_bytes(b'{"d_01":{"01":"\\ud800"}}')
        result = run_validation(self.data_path, self.pred_path, split="dev")
        self.assert_error(result, "INVALID_UNICODE_SCALAR")
        json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.data_path.write_bytes(b'{"d":{"dialogue":[{"sentence_id":"1","sentence":"a","BIO_label":"O","\\ud800":0}]}}')
        result = run_validation(self.data_path, self.pred_path, split="dev")
        self.assert_error(result, "INVALID_UNICODE_SCALAR")
        json.dumps(result, ensure_ascii=False).encode("utf-8")

    def test_invalid_data_json_and_root_including_null(self):
        self.write_json(self.pred_path, self.pred)
        for raw, code in ((b'{', "INVALID_JSON"), (b'null', "DATA_ROOT_TYPE"),
                          (b'[]', "DATA_ROOT_TYPE"), (b'{}', "DATA_ROOT_TYPE")):
            with self.subTest(raw=raw):
                self.data_path.write_bytes(raw)
                self.assert_error(run_validation(self.data_path, self.pred_path, split="dev"), code)

    def test_data_sentence_schema_errors(self):
        for field, value, code in (("sentence_id", 1, "DATA_SENTENCE_ID_TYPE"),
                                   ("sentence", [], "DATA_TEXT_TYPE"),
                                   ("BIO_label", None, "MISSING_OR_INVALID_GOLD")):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data["d_01"]["dialogue"][0][field] = value
                self.assert_error(self.validate(data=data), code)

    def test_bad_gold_alignment_and_bio(self):
        for sequence, code in (("O", "LABEL_LENGTH_MISMATCH"),
                                ("I-Symptom O", "ILLEGAL_I_TRANSITION"),
                                ("PAD O", "INVALID_LABEL"), ("O  O", "INVALID_SEPARATOR")):
            with self.subTest(sequence=sequence):
                self.data["d_01"]["dialogue"][0]["BIO_label"] = sequence
                self.assert_error(self.validate(), code)

    def test_raw_test_has_text_no_gold(self):
        for dialogue in self.data.values():
            for row in dialogue["dialogue"]:
                del row["BIO_label"]
        result = self.validate(split="test")
        self.assertEqual("PASS", result["status"])
        self.assertEqual("PASS", result["checks"]["prediction_alignment"]["status"])
        self.assertEqual(0, result["data"]["sentences_with_gold"])

    def test_test_unexpected_gold(self):
        self.assert_error(self.validate(split="test"), "UNEXPECTED_TEST_GOLD")

    def test_template_cannot_claim_alignment_pass(self):
        template = {did: {sid: "" for sid in rows} for did, rows in self.pred.items()}
        result = self.validate(data=template, split="test")
        self.assertEqual("NOT_RUN", result["status"])
        self.assertEqual("PASS", result["checks"]["id_coverage"]["status"])
        self.assertEqual("NOT_RUN", result["checks"]["prediction_alignment"]["status"])
        self.assertEqual(3, result["predictions"]["expected_sentences"]["not_fully_checked"])
        self.pred["d_01"]["01"] = "I-Drug"
        self.assert_error(self.validate(data=template, split="test"), "ILLEGAL_I_TRANSITION")

    def test_missing_text_in_raw_test_fails(self):
        del self.data["d_01"]["dialogue"][0]["sentence"]
        for d in self.data.values():
            for r in d["dialogue"]:
                del r["BIO_label"]
        self.assert_error(self.validate(split="test"), "DATA_TEXT_TYPE")

    def test_directory_and_nested_directory_resolution(self):
        self.write_json(self.pred_path, self.pred)
        for nested in (False, True):
            with self.subTest(nested=nested):
                base = self.folder / str(nested)
                actual = base / "IMCS-V2-NER" if nested else base
                actual.mkdir(parents=True)
                self.write_json(actual / "IMCS-V2_dev.json", self.data)
                self.assertEqual("PASS", run_validation(base, self.pred_path, split="dev")["status"])

    def test_zip_readonly_and_source_hashes(self):
        archive_path = self.folder / "data.zip"
        raw = json.dumps(self.data, ensure_ascii=False).encode("utf-8")
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("IMCS-V2-NER/IMCS-V2_dev.json", raw)
        before = archive_path.read_bytes()
        self.write_json(self.pred_path, self.pred)
        result = run_validation(archive_path, self.pred_path, split="dev")
        self.assertEqual("PASS", result["status"])
        self.assertEqual(before, archive_path.read_bytes())
        self.assertEqual(hashlib.sha256(before).hexdigest(), result["inputs"]["data"]["archive_sha256"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), result["inputs"]["data"]["sha256"])

    def test_test_zip_prefers_original_text_and_falls_back_explicitly(self):
        for d in self.data.values():
            for row in d["dialogue"]:
                del row["BIO_label"]
        template = {did: {sid: "" for sid in rows} for did, rows in self.pred.items()}
        self.write_json(self.pred_path, self.pred)
        for has_raw in (True, False):
            with self.subTest(has_raw=has_raw):
                archive_path = self.folder / (str(has_raw) + ".zip")
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("IMCS-V2-NER/IMCS-V2-NER_test.json", json.dumps(template))
                    if has_raw:
                        archive.writestr("IMCS-V2-NER/IMCS-V2_test.json", json.dumps(self.data))
                result = run_validation(archive_path, self.pred_path, split="test")
                self.assertEqual("PASS" if has_raw else "NOT_RUN", result["status"])
                self.assertEqual("raw" if has_raw else "empty_test_template", result["data"]["format"])

    def test_ambiguous_directory_and_zip(self):
        self.write_json(self.pred_path, self.pred)
        self.write_json(self.folder / "IMCS-V2_dev.json", self.data)
        nested = self.folder / "IMCS-V2-NER"
        nested.mkdir()
        self.write_json(nested / "IMCS-V2_dev.json", self.data)
        self.assert_error(run_validation(self.folder, self.pred_path, split="dev"), "AMBIGUOUS_DATA")
        archive_path = self.folder / "ambiguous.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("a/IMCS-V2_dev.json", "{}")
            archive.writestr("b/IMCS-V2_dev.json", "{}")
        self.assert_error(run_validation(archive_path, self.pred_path, split="dev"), "AMBIGUOUS_DATA")

    def test_split_mismatch(self):
        self.write_json(self.folder / "IMCS-V2_train.json", self.data)
        self.write_json(self.pred_path, self.pred)
        self.assert_error(run_validation(self.folder / "IMCS-V2_train.json", self.pred_path, split="dev"),
                          "SPLIT_MISMATCH")

    def test_missing_files_and_bad_zip(self):
        self.write_json(self.pred_path, self.pred)
        self.assert_error(run_validation(self.data_path, self.pred_path, split="dev"), "DATA_READ_ERROR")
        self.write_json(self.data_path, self.data)
        self.pred_path.unlink()
        self.assert_error(run_validation(self.data_path, self.pred_path, split="dev"), "PREDICTION_READ_ERROR")
        bad_zip = self.folder / "bad.zip"
        bad_zip.write_bytes(b"not a ZIP")
        self.assert_error(run_validation(bad_zip, self.pred_path, split="dev"), "DATA_READ_ERROR")

    def test_error_counts_exhaustive_examples_bounded_no_text(self):
        text = "SYNTHETIC_PRIVATE_SENTENCE"
        data = {"d": {"dialogue": [{"sentence_id": str(i), "sentence": text,
                                      "BIO_label": " ".join(["O"] * len(text))} for i in range(100)]}}
        pred = {"d": {str(i): "INVALID" for i in range(100)}}
        result = self.validate(data=data, pred=pred, max_examples=4)
        self.assertEqual(100, result["errors"]["by_code"]["INVALID_LABEL"])
        self.assertEqual(100, result["errors"]["by_code"]["LABEL_LENGTH_MISMATCH"])
        self.assertEqual(4, len(result["errors"]["examples"]))
        self.assertEqual(196, result["errors"]["examples_omitted"])
        self.assertEqual(100, result["predictions"]["expected_sentences"]["failed"])
        self.assertNotIn(text, json.dumps(result))
        self.assertEqual([], self.validate(data=data, pred=pred, max_examples=0)["errors"]["examples"])

    def test_cli_exit_codes_and_reports(self):
        self.validate()
        report_path = self.folder / "report.json"
        args = ("--data", self.data_path, "--predictions", self.pred_path,
                "--split", "dev", "--report", report_path)
        process = self.cli(*args)
        self.assertEqual(0, process.returncode, process.stderr)
        self.assertEqual("PASS", json.loads(report_path.read_text(encoding="utf-8"))["status"])
        self.pred_path.write_bytes(b"null")
        self.assertEqual(1, self.cli(*args).returncode)
        template = {did: {sid: "" for sid in rows} for did, rows in self.pred.items()}
        self.write_json(self.data_path, template)
        self.write_json(self.pred_path, self.pred)
        process = self.cli("--data", self.data_path, "--predictions", self.pred_path,
                           "--split", "test", "--report", report_path)
        self.assertEqual(2, process.returncode, process.stderr)
        self.assertEqual("NOT_RUN", json.loads(report_path.read_text(encoding="utf-8"))["status"])

    def test_cli_audit_only_and_overwrite_protection(self):
        self.validate()
        report_path = self.folder / "audit.json"
        process = self.cli("--data", self.data_path, "--split", "dev", "--audit-data", "--report", report_path)
        self.assertEqual(0, process.returncode, process.stderr)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual("data_audit", report["mode"])
        self.assertIsNone(report["predictions"])
        before = self.data_path.read_bytes()
        self.assertEqual(2, self.cli("--data", self.data_path, "--split", "dev", "--audit-data",
                                     "--report", self.data_path).returncode)
        self.assertEqual(before, self.data_path.read_bytes())
        self.assertEqual(2, self.cli("--data", self.folder, "--split", "dev", "--audit-data",
                                     "--report", report_path).returncode)
        before = self.pred_path.read_bytes()
        self.assertEqual(2, self.cli("--data", self.data_path, "--predictions", self.pred_path,
                                     "--split", "dev", "--report", self.pred_path).returncode)
        self.assertEqual(before, self.pred_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
