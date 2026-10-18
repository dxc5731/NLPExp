"""Artificial stage-three handoffs only. No actual models or inferred causes."""
import copy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.check_run_info import check_run_info, OFFICIAL_ARCHIVE_SHA256
from src.compare_evaluations import compare_evaluations, verify_comparison_files
from src.evaluate_predictions import evaluate
from src.evaluation_io import read_json, sha256, write_csv, new_directory
from src.export_errors import build_details, candidate_links, export_errors, verify_error_files
from src.ner_metrics import Entity
from src.run_evaluation_batch import run_batch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/pipeline_validation"


class PipelineFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="中文 接入测试_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = self.root / "人工 交接"
        shutil.copytree(FIXTURE, self.fixture, ignore=shutil.ignore_patterns("__pycache__"))
        self.manifest = self.fixture / "evaluation_runs.json"
        self.info = read_json(self.manifest)
        self.output = self.root / "独立 结果"

    def write(self, path, value):
        Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def save_manifest(self):
        self.write(self.manifest, self.info)

    def batch(self, **kwargs):
        self.save_manifest()
        return run_batch(self.manifest, output_dir=self.output, **kwargs)

    def eval_one(self, name="crf", purpose="synthetic_integration_test"):
        return evaluate(self.fixture / "data_dev.json", self.fixture / name / "predictions_dev.json",
                        purpose=purpose, return_inputs=True)

    def metadata(self, name="crf", result=None):
        if result is None:
            result = self.eval_one(name)[0]
        index = {"rules": 0, "crf": 1, "bert": 2}[name]
        row = self.info["runs"][index]
        return check_run_info(self.fixture / name / "run_info.json", result,
                              run_id=row["run_id"], method_name=row["method_name"])

    def change_info(self, name, edit):
        path = self.fixture / name / "run_info.json"
        info = read_json(path)
        edit(info)
        self.write(path, info)

    def mark_real_style_fixture(self):
        # Exercises the actual-metadata schema in TEMP only. These remain unit
        # test declarations, never a claim that any model/inference really ran.
        path = self.fixture / "rules/run_info.json"
        info = read_json(path)
        info.update(artifact_kind="model_prediction", device="unit-test CPU declaration; no inference",
                    inference_command="NOT_EXECUTED_UNIT_TEST", inference_working_directory=str(self.fixture),
                    inference_elapsed_seconds=0.125, timing_scope="synthetic metadata type test; not a measurement",
                    training_applicability="No training in this schema test", seed_applicability="deterministic fixture")
        info["data"]["archive_sha256"] = OFFICIAL_ARCHIVE_SHA256
        self.write(path, info)
        return info


class ErrorExportTests(PipelineFixture):
    def test_fp_fn_tp_records_equal_evaluation(self):
        result, data, pred = self.eval_one()
        errors, matches, candidates, summary, _ = build_details(result, data, pred, run_id="r", method_name="人工")
        self.assertEqual((5, 2, 3), (len(errors), len(matches), len(candidates)))
        self.assertEqual({"FN": 3, "FP": 2, "TP": 2}, summary["rows"])
        self.assertEqual(result["entities"], summary["entities"])
        self.assertEqual({"type_mismatch": 1, "boundary_mismatch": 1, "suspected_miss": 1}, summary["candidate_counts"])
        self.assertEqual(7, len({r["record_id"] for r in errors + matches}))
        self.assertTrue(all(r["gold_sha256"] == result["inputs"]["data"]["sha256"] for r in errors))

    def test_type_candidate_exact_same_span(self):
        p, g = Entity("d", "s", "Drug", 1, 3), Entity("d", "s", "Symptom", 1, 3)
        self.assertEqual([("type_mismatch", p, g, "unique_one_to_one")], candidate_links({p}, {g}))

    def test_boundary_candidate_same_type_overlap(self):
        p, g = Entity("d", "s", "Drug", 1, 3), Entity("d", "s", "Drug", 2, 4)
        self.assertEqual([("boundary_mismatch", p, g, "unique_one_to_one")], candidate_links({p}, {g}))

    def test_ambiguous_one_to_many_remains_unlinked(self):
        p = Entity("d", "s", "Drug", 0, 4)
        gold = {Entity("d", "s", "Drug", 0, 2), Entity("d", "s", "Drug", 2, 4)}
        candidates = candidate_links({p}, gold)
        self.assertEqual(3, len(candidates))
        self.assertTrue(all(row[3] == "ambiguous_unlinked" for row in candidates))
        self.assertFalse(any(row[0] == "boundary_mismatch" for row in candidates))

    def test_touching_spans_not_boundary_pair(self):
        rows = candidate_links({Entity("d", "s", "Drug", 0, 1)}, {Entity("d", "s", "Drug", 1, 2)})
        self.assertEqual(["suspected_spurious", "suspected_miss"], [r[0] for r in rows])

    def test_candidates_deterministic_no_repeated_pair(self):
        result, data, pred = self.eval_one()
        first = build_details(result, data, pred, run_id="r", method_name="m")
        second = build_details(result, data, pred, run_id="r", method_name="m")
        self.assertEqual(first, second)
        linked = [r[k] for r in first[2] for k in ("fp_record_id", "fn_record_id") if r[k]]
        self.assertEqual(5, len(linked))
        self.assertEqual(len(linked), len(set(linked)))

    def test_count_mismatch_refuses_export_before_creating_files(self):
        result, data, pred = self.eval_one()
        wrong = copy.deepcopy(pred)
        wrong["manual_scoring"]["01"] = " ".join(["O"] * 15)
        path = self.root / "bad_errors"
        with self.assertRaises(ValueError):
            export_errors(result, data, wrong, path, run_id="r", method_name="m")
        self.assertFalse(path.exists())

    def test_reopened_csv_count_tampering_detected(self):
        result, data, pred = self.eval_one()
        target = self.root / "errors"
        export_errors(result, data, pred, target, run_id="r", method_name="m")
        path = target / "entity_errors.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_error_files(result, target)

    def test_default_no_medical_text_local_export_explicit(self):
        result, data, pred = self.eval_one("bert")
        target = self.root / "default"
        export_errors(result, data, pred, target, run_id="r", method_name="m")
        self.assertFalse((target / "local_analysis").exists())
        for path in target.glob("*"):
            self.assertNotIn("甲乙，丙丁", path.read_text(encoding="utf-8"))
        local = self.root / "private"
        export_errors(result, data, pred, local, run_id="r", method_name="m", local_text=True)
        rows = [json.loads(s) for s in (local / "local_analysis/sentences.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(2, len(rows))
        self.assertEqual("A 😀，𠀀é", rows[1]["sentence"])
        self.assertEqual("A", rows[1]["predicted_entities"][0]["text"])

    def test_csv_formula_defense_and_bom(self):
        path = self.root / "safe.csv"
        write_csv(path, ["value"], [{"value": x} for x in ("=1+1", "+x", " -2", "@X", "\tX", "普通中文")], bom=True)
        self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
        with path.open(encoding="utf-8-sig", newline="") as stream:
            values = [r["value"] for r in csv.DictReader(stream)]
        self.assertEqual(["'=1+1", "'+x", "' -2", "'@X", "'\tX", "普通中文"], values)


class MetadataTests(PipelineFixture):
    def test_valid_synthetic_metadata_and_not_applicable(self):
        report = self.metadata()
        self.assertEqual(("PASS", "PASS"), (report["automated_status"], report["status"]))
        self.assertIn("NOT_APPLICABLE", {r["status"] for r in report["checks"]})
        self.assertIsNone(report["inference_elapsed_seconds_for_comparison"])

    def test_missing_required_metadata_fields_reported(self):
        self.change_info("crf", lambda r: r.pop("method_version"))
        report = self.metadata()
        self.assertEqual("FAIL", report["status"])
        self.assertIn("method_version", [r["name"] for r in report["checks"] if r["status"] == "FAIL"])

    def test_metadata_identity_mismatch(self):
        self.change_info("crf", lambda r: r.update(run_id="wrong"))
        self.assertEqual("FAIL", self.metadata()["automated_status"])

    def test_prediction_hash_mismatch(self):
        self.change_info("crf", lambda r: r["prediction"].update(sha256="0" * 64))
        self.assertEqual("FAIL", self.metadata()["status"])

    def test_config_hash_checked_even_with_inline_config(self):
        (self.fixture / "crf/config.json").write_text("{}", encoding="utf-8")
        self.change_info("crf", lambda r: r["config"].update(path="config.json", sha256="0" * 64))
        self.assertEqual("FAIL", self.metadata()["status"])

    def test_missing_log_or_snapshot_rejected(self):
        (self.fixture / "crf/fixture_creation.txt").unlink()
        (self.fixture / "fixture_source.txt").unlink()
        failed = [r["name"] for r in self.metadata()["checks"] if r["status"] == "FAIL"]
        self.assertIn("code.snapshot", failed)
        self.assertIn("logs[0]", failed)

    def test_missing_metadata_readable_failure_report(self):
        report = check_run_info(None, self.eval_one()[0], run_id="r", method_name="m")
        self.assertEqual("FAIL", report["status"])
        self.assertEqual("run_info_read", report["checks"][0]["name"])

    def test_real_style_fields_do_not_prove_actual_reproducibility(self):
        self.mark_real_style_fixture()
        result = self.eval_one("rules", purpose="model_evaluation")[0]
        report = self.metadata("rules", result)
        self.assertEqual(("PASS", "NEEDS_REVIEW"), (report["automated_status"], report["status"]))
        pending = {r["name"] for r in report["checks"] if r["status"] == "NEEDS_REVIEW"}
        self.assertIn("actual_model_identity", pending)
        self.assertIn("absence_of_data_leakage", pending)
        self.assertIn("truth_of_inference_timing", pending)
        self.assertIsNone(report["inference_elapsed_seconds_for_comparison"])

    def test_timing_only_displayed_with_hashed_human_attestation(self):
        info = self.mark_real_style_fixture()
        review = self.fixture / "rules/review.txt"
        review.write_text("Artificial schema test, no real human measurement claim.", encoding="utf-8")
        info["manual_review"] = {"timing_verified": True, "reviewer": "UNIT_TEST_ONLY",
                                 "record": {"path": "review.txt", "sha256": sha256(review)}}
        self.write(self.fixture / "rules/run_info.json", info)
        report = self.metadata("rules", self.eval_one("rules", purpose="model_evaluation")[0])
        self.assertEqual("NEEDS_REVIEW", report["status"])
        self.assertEqual(0.125, report["inference_elapsed_seconds_for_comparison"])

    def test_bert_alignment_missing_structured_fields(self):
        self.change_info("bert", lambda r: r["alignment"].pop("long_text"))
        self.assertEqual("FAIL", self.metadata("bert")["status"])

    def test_synthetic_cannot_be_relabelled_as_real_comparison(self):
        report = self.metadata("crf", self.eval_one("crf", purpose="model_evaluation")[0])
        self.assertEqual("FAIL", report["automated_status"])
        self.assertIn("artifact_kind", [r["name"] for r in report["checks"] if r["status"] == "FAIL"])

    def test_learned_metadata_requires_training_records_and_seeds(self):
        self.mark_real_style_fixture()
        self.change_info("rules", lambda r: r["method"].update(kind="crf"))
        result = self.eval_one("rules", purpose="model_evaluation")[0]
        self.assertEqual("FAIL", self.metadata("rules", result)["status"])


class BatchComparisonTests(PipelineFixture):
    def test_relabelling_fixture_as_real_does_not_bypass_official_gold_check(self):
        self.mark_real_style_fixture()
        self.info["purpose"] = "model_evaluation"
        self.info["runs"] = self.info["runs"][:1]
        report = self.batch()
        self.assertEqual("PARTIAL_SUCCESS", report["status"])
        self.assertEqual("SCORED", report["runs"][0]["status"])
        self.assertEqual(0, report["comparison"]["included"])
        self.assertIn("OFFICIAL_DATA_MISMATCH", report["comparison"]["excluded"][0]["reason"])

    def test_saved_comparison_csvs_equal_independent_metrics(self):
        report = self.batch()
        directory = Path(report["comparison"]["path"]).parent
        comparison = read_json(directory / "comparison.json")
        self.assertTrue(verify_comparison_files(comparison, directory))
        independent = {r["run_id"]: read_json(r["evaluation_path"]) for r in report["runs"]}
        with (directory / "model_comparison.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                result = independent[row["run_id"]]
                self.assertEqual(result["entities"]["micro"]["f1"], float(row["entity_micro_f1"]))
                self.assertEqual(result["characters"]["accuracy"], float(row["character_accuracy"]))
        with (directory / "entity_type_comparison.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(15, len(rows))
        for row in rows:
            metric = independent[row["run_id"]]["entities"]["per_type"][row["entity_type"]]
            for key in ("tp", "fp", "fn", "support"):
                self.assertEqual(metric[key], int(row[key]))
            for key in ("precision", "recall", "f1"):
                self.assertEqual(metric[key], float(row[key]))

    def test_saved_comparison_cell_tampering_detected(self):
        report = self.batch()
        directory = Path(report["comparison"]["path"]).parent
        comparison = read_json(directory / "comparison.json")
        path = directory / "model_comparison.csv"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("0.8", "0.99", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "CSV cells differ"):
            verify_comparison_files(comparison, directory)

    def test_invalid_package_version_metadata_is_debug_only(self):
        self.change_info("crf", lambda r: r["environment"].update(packages={"framework": None}))
        report = self.batch()
        self.assertEqual("DEBUG_SCORED", report["runs"][1]["status"])
        self.assertEqual(2, report["comparison"]["included"])

    def test_partial_batch_cli_returns_two_and_keeps_other_runs(self):
        (self.fixture / "crf/predictions_dev.json").write_text("null", encoding="utf-8")
        self.info["output_dir"] = str(self.output)
        self.save_manifest()
        process = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "src/run_evaluation_batch.py"),
                                  "--manifest", str(self.manifest)], capture_output=True, encoding="utf-8")
        self.assertEqual(2, process.returncode, process.stdout + process.stderr)
        path, = self.output.glob("*/batch_report.json")
        report = read_json(path)
        self.assertEqual("PARTIAL_SUCCESS", report["status"])
        self.assertEqual(["SCORED", "REJECTED", "SCORED"], [r["status"] for r in report["runs"]])

    def test_three_method_end_to_end_exact_counts_and_metrics(self):
        report = self.batch()
        self.assertEqual(("SUCCESS", "PASS"), (report["status"], report["self_check"]))
        self.assertEqual(3, report["comparison"]["included"])
        expected = [(0, 0, 5, 0.0), (2, 2, 3, 4 / 9), (4, 1, 1, 0.8)]
        comparison = read_json(report["comparison"]["path"])
        by_run = {r["run"]["run_id"]: r for r in comparison["included"]}
        for outcome, numbers in zip(report["runs"], expected):
            result = read_json(outcome["evaluation_path"])
            micro = result["entities"]["micro"]
            self.assertEqual(numbers[:3], (micro["tp"], micro["fp"], micro["fn"]))
            self.assertAlmostEqual(numbers[3], micro["f1"], delta=1e-12)
            self.assertEqual(result["entities"], by_run[outcome["run_id"]]["entities"])
            self.assertEqual(result["characters"], by_run[outcome["run_id"]]["characters"])
            self.assertTrue(verify_error_files(result, Path(outcome["evaluation_path"]).parent / "errors"))
        self.assertEqual(15, len(comparison["entity_types"]))
        self.assertEqual([0.8, 4 / 9, 0.0], [r["entity_micro_f1"] for r in comparison["overview"]])
        self.assertTrue(all(r["purpose"] == "synthetic_integration_test" for r in comparison["overview"]))

    def test_batch_metrics_equal_independent_scorer(self):
        report = self.batch()
        for outcome, name in zip(report["runs"], ("rules", "crf", "bert")):
            standalone = self.eval_one(name)[0]
            batch = read_json(outcome["evaluation_path"])
            self.assertEqual(standalone["entities"], batch["entities"])
            self.assertEqual(standalone["characters"], batch["characters"])

    def test_missing_metadata_keeps_debug_scores_and_excludes_comparison(self):
        (self.fixture / "crf/run_info.json").unlink()
        report = self.batch()
        self.assertEqual("PARTIAL_SUCCESS", report["status"])
        self.assertEqual("DEBUG_SCORED", report["runs"][1]["status"])
        self.assertTrue(Path(report["runs"][1]["evaluation_path"]).exists())
        self.assertEqual(2, report["comparison"]["included"])
        self.assertEqual(1, len(report["comparison"]["excluded"]))

    def test_invalid_prediction_does_not_lose_other_runs(self):
        (self.fixture / "crf/predictions_dev.json").write_text("null", encoding="utf-8")
        report = self.batch()
        self.assertEqual("PARTIAL_SUCCESS", report["status"])
        self.assertEqual(["SCORED", "REJECTED", "SCORED"], [r["status"] for r in report["runs"]])
        rejection = read_json(report["runs"][1]["rejection_path"])
        self.assertNotIn("entities", rejection)
        self.assertEqual(2, report["comparison"]["included"])

    def test_duplicate_run_ids_recorded_not_silently_dropped(self):
        self.info["runs"][1]["run_id"] = self.info["runs"][0]["run_id"]
        report = self.batch()
        self.assertEqual(3, len(report["runs"]))
        self.assertEqual(["FAILED", "FAILED", "SCORED"], [r["status"] for r in report["runs"]])
        self.assertIn("DUPLICATE_RUN_ID", report["runs"][0]["reason"])
        self.assertEqual("PARTIAL_SUCCESS", report["status"])

    def test_all_invalid_runs_overall_failure(self):
        for name in ("rules", "crf", "bert"):
            (self.fixture / name / "predictions_dev.json").unlink()
        report = self.batch()
        self.assertEqual("FAILURE", report["status"])
        self.assertEqual(3, len(report["runs"]))
        self.assertEqual(0, report["comparison"]["included"])

    def test_one_arrival_is_enough(self):
        self.info["runs"] = self.info["runs"][:1]
        report = self.batch()
        self.assertEqual("SUCCESS", report["status"])
        self.assertEqual(1, report["comparison"]["included"])

    def test_same_method_two_distinct_run_ids_preserved(self):
        self.info["runs"] = self.info["runs"][:1]
        duplicate = copy.deepcopy(self.info["runs"][0])
        duplicate["run_id"] = "synthetic_rules_second"
        copied = self.fixture / "rules_second"
        shutil.copytree(self.fixture / "rules", copied)
        metadata = read_json(copied / "run_info.json")
        metadata["run_id"] = duplicate["run_id"]
        self.write(copied / "run_info.json", metadata)
        duplicate.update(predictions="rules_second/predictions_dev.json", run_info="rules_second/run_info.json")
        self.info["runs"].append(duplicate)
        report = self.batch()
        self.assertEqual("SUCCESS", report["status"])
        self.assertEqual(2, report["comparison"]["included"])
        self.assertEqual(2, len({r["evaluation_path"] for r in report["runs"]}))

    def test_data_version_mismatch_refuses_combined_table(self):
        self.change_info("crf", lambda r: r["data"].update(version="other-version"))
        report = self.batch()
        self.assertEqual("INCOMPATIBLE", report["comparison"]["status"])
        self.assertEqual(0, report["comparison"]["included"])
        self.assertEqual("PARTIAL_SUCCESS", report["status"])

    def test_gold_content_hash_mismatch_refuses_combined_table(self):
        source = read_json(self.fixture / "data_dev.json")
        source["manual_scoring"]["extra_synthetic_field"] = True
        self.write(self.fixture / "other_dev.json", source)
        self.info["runs"][1]["data"] = {"path": "other_dev.json", "split": "dev"}
        self.change_info("crf", lambda r: r["data"].update(source_file="../other_dev.json", source_sha256=sha256(self.fixture / "other_dev.json")))
        report = self.batch()
        self.assertEqual("INCOMPATIBLE", report["comparison"]["status"])
        self.assertEqual(0, report["comparison"]["included"])

    def test_split_metadata_mismatch_excluded(self):
        self.change_info("crf", lambda r: r["data"].update(split="test"))
        report = self.batch()
        self.assertEqual("DEBUG_SCORED", report["runs"][1]["status"])
        self.assertEqual(2, report["comparison"]["included"])

    def test_test_without_gold_rejected(self):
        source = read_json(self.fixture / "data_dev.json")
        for row in source["manual_scoring"]["dialogue"]:
            del row["BIO_label"]
        self.write(self.fixture / "test.json", source)
        self.info["runs"][1]["data"] = {"path": "test.json", "split": "test"}
        report = self.batch()
        self.assertEqual("PARTIAL_SUCCESS", report["status"])
        self.assertEqual("TEST_HAS_NO_PUBLIC_GOLD", report["runs"][1]["reason"]["code"])

    def test_synthetic_results_cannot_enter_real_comparison(self):
        report = self.batch()
        comparison = compare_evaluations([r["evaluation_path"] for r in report["runs"]], purpose="model_evaluation")
        self.assertEqual("NO_COMPARABLE_RESULTS", comparison["status"])
        self.assertEqual(3, len(comparison["excluded"]))
        self.assertEqual([], comparison["overview"])

    def test_mixed_purpose_manifest_entry_rejected(self):
        self.info["runs"][1]["purpose"] = "model_evaluation"
        report = self.batch()
        self.assertEqual("FAILED", report["runs"][1]["status"])
        self.assertEqual("PARTIAL_SUCCESS", report["status"])

    def test_changed_metric_policy_refuses_combination(self):
        report = self.batch()
        paths = [r["evaluation_path"] for r in report["runs"]]
        result = read_json(paths[1])
        result["metric_policy"]["span"] = "different coordinates"
        self.write(paths[1], result)
        comparison = compare_evaluations(paths, purpose="synthetic_integration_test")
        self.assertEqual("INCOMPATIBLE", comparison["status"])
        self.assertEqual([], comparison["overview"])

    def test_tampered_metrics_excluded(self):
        report = self.batch()
        paths = [r["evaluation_path"] for r in report["runs"]]
        result = read_json(paths[1])
        result["entities"]["micro"]["f1"] = 0.999
        self.write(paths[1], result)
        comparison = compare_evaluations(paths, purpose="synthetic_integration_test")
        self.assertEqual(1, len(comparison["excluded"]))
        self.assertEqual(2, len(comparison["included"]))

    def test_modified_prediction_hash_prevents_stale_comparison(self):
        report = self.batch()
        (self.fixture / "crf/predictions_dev.json").write_text("{}", encoding="utf-8")
        comparison = compare_evaluations([r["evaluation_path"] for r in report["runs"]], purpose="synthetic_integration_test")
        self.assertEqual(2, len(comparison["included"]))
        self.assertIn("prediction hash changed", comparison["excluded"][0]["reason"])

    def test_repeat_batch_creates_unique_directories_and_preserves_inputs(self):
        original = {str(p): p.read_bytes() for p in self.fixture.rglob("*") if p.is_file()}
        first = self.batch()
        path = Path(first["directory"]) / "batch_report.json"
        previous = path.read_bytes()
        second = self.batch()
        self.assertNotEqual(first["directory"], second["directory"])
        self.assertEqual(previous, path.read_bytes())
        for name, content in original.items():
            self.assertEqual(content, Path(name).read_bytes())

    def test_paths_relative_to_manifest_not_working_directory(self):
        self.info["output_dir"] = str(self.output)
        self.save_manifest()
        process = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "src/run_evaluation_batch.py"),
                                  "--manifest", str(self.manifest)], cwd=self.root, capture_output=True, encoding="utf-8")
        self.assertEqual(0, process.returncode, process.stderr + process.stdout)
        self.assertIn("SUCCESS", process.stdout)
        self.assertEqual(1, len(list(self.output.glob("*/batch_report.json"))))

    def test_windows_absolute_paths(self):
        for row in self.info["runs"]:
            row["predictions"] = str((self.fixture / row["predictions"]).resolve())
            row["run_info"] = str((self.fixture / row["run_info"]).resolve())
        self.info["data"]["path"] = str((self.fixture / "data_dev.json").resolve())
        self.assertEqual("SUCCESS", self.batch()["status"])

    def test_output_does_not_descend_from_input_file_or_data_directory(self):
        self.save_manifest()
        with self.assertRaises(ValueError):
            run_batch(self.manifest, output_dir=self.fixture / "crf/predictions_dev.json")
        data_dir = self.fixture / "data"
        data_dir.mkdir()
        shutil.copy(self.fixture / "data_dev.json", data_dir / "IMCS-V2_dev.json")
        self.info["data"]["path"] = "data"
        self.save_manifest()
        with self.assertRaises(ValueError):
            run_batch(self.manifest, output_dir=data_dir / "outputs")
        self.assertFalse((data_dir / "outputs").exists())

    def test_final_saved_error_self_check_failure_is_visible(self):
        from src.export_errors import verify_error_files as real_check
        calls = 0

        def fail_final(result, directory):
            nonlocal calls
            calls += 1
            # Batch final self-check calls its own imported alias after exports.
            if calls == 1:
                raise ValueError("injected count mismatch")
            return real_check(result, directory)

        with patch("src.run_evaluation_batch.verify_error_files", side_effect=fail_final):
            report = self.batch()
        self.assertEqual("FAIL", report["self_check"])
        self.assertEqual("FAILED", report["runs"][0]["status"])
        self.assertEqual("INVALIDATED_BY_FINAL_SELF_CHECK", report["comparison"]["status"])
        self.assertEqual(0, report["comparison"]["included"])


if __name__ == "__main__":
    unittest.main()
