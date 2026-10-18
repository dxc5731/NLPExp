"""Repeat stage-three acceptance; all scores use explicitly artificial fixtures."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate_predictions import evaluate
from src.evaluation_io import new_directory, read_json, sha256, write_json
from src.export_errors import verify_error_files
from src.compare_evaluations import verify_comparison_files
from src.check_run_info import OFFICIAL_ARCHIVE_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-data", type=Path, default=ROOT / "data/IMCS-V2-NER.zip")
    args = parser.parse_args()
    directory = new_directory(ROOT / "logs/pipeline_validation/acceptance")
    fixture = ROOT / "examples/pipeline_validation"
    before = {str(p): sha256(p) for p in fixture.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    official_before = sha256(args.official_data)
    assert official_before == OFFICIAL_ARCHIVE_SHA256, "Unexpected official archive version"
    evidence = {"purpose": "synthetic_integration_test", "commands": [], "test_counts": {},
                "notice": "Artificial handoff metrics only; official dev is read-only audited, never scored here."}

    def execute(name, arguments):
        command = [sys.executable, "-X", "utf8", *map(str, arguments)]
        start = perf_counter()
        process = subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8", errors="strict")
        elapsed = perf_counter() - start
        output = process.stdout + process.stderr
        transcript = directory / (name + ".txt")
        with transcript.open("x", encoding="utf-8") as stream:
            stream.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
            stream.write(f"EXIT: {process.returncode}\nELAPSED_SECONDS: {elapsed}\n" + output)
        evidence["commands"].append({"name": name, "arguments": command, "exit_code": process.returncode,
                                     "elapsed_seconds": elapsed, "transcript": str(transcript)})
        print(name, "exit=", process.returncode, "seconds=", round(elapsed, 3), flush=True)
        assert process.returncode == 0, output
        return output

    for stage, pattern in (("stage1", "test_validate_predictions.py"), ("stage2", "test_evaluate_predictions.py"),
                           ("stage3", "test_evaluation_pipeline.py"), ("all", "test*.py")):
        output = execute(stage, ["-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"])
        count = int(re.search(r"Ran (\d+) tests", output).group(1))
        evidence["test_counts"][stage] = count
    assert evidence["test_counts"]["stage1"] == 42
    assert evidence["test_counts"]["stage2"] == 47
    assert evidence["test_counts"]["all"] == sum(evidence["test_counts"][s] for s in ("stage1", "stage2", "stage3"))
    execute("three_method_cli", ["src/run_evaluation_batch.py", "--manifest", fixture / "evaluation_runs.json",
                                "--output-dir", directory / "batches", "--csv-bom"])
    batch_path, = (directory / "batches").glob("*/batch_report.json")
    batch = read_json(batch_path)
    assert (batch["status"], batch["self_check"]) == ("SUCCESS", "PASS")
    comparison = read_json(batch["comparison"]["path"])
    assert len(comparison["included"]) == 3
    verify_comparison_files(comparison, Path(batch["comparison"]["path"]).parent)
    expected = [(0, 0, 5, 0.0), (2, 2, 3, 4 / 9), (4, 1, 1, 0.8)]
    evidence["artificial_runs"] = []
    for outcome, name, numbers in zip(batch["runs"], ("rules", "crf", "bert"), expected):
        result = read_json(outcome["evaluation_path"])
        independent = evaluate(fixture / "data_dev.json", fixture / name / "predictions_dev.json",
                               split="dev", purpose="synthetic_integration_test")
        assert independent["entities"] == result["entities"]
        assert independent["characters"] == result["characters"]
        micro = result["entities"]["micro"]
        assert tuple(micro[k] for k in ("tp", "fp", "fn")) == numbers[:3]
        assert abs(micro["f1"] - numbers[3]) <= 1e-12
        verify_error_files(result, Path(outcome["evaluation_path"]).parent / "errors")
        assert result["counts"] == {"dialogues": 1, "sentences": 2, "characters": 22}
        evidence["artificial_runs"].append({"run_id": outcome["run_id"], "entity_micro": micro,
                                            "independent_metrics_match": True, "saved_errors_match": True})
    audit_path = directory / "official_dev_readonly.json"
    execute("official_dev_readonly", ["src/validate_predictions.py", "--data", args.official_data,
                                      "--split", "dev", "--audit-data", "--report", audit_path])
    audit = read_json(audit_path)
    assert audit["status"] == "PASS" and audit["errors"]["total"] == 0
    assert tuple(audit["data"][k] for k in ("dialogues", "sentences", "characters")) == (833, 33267, 441339)
    assert sha256(args.official_data) == official_before
    assert all(sha256(path) == digest for path, digest in before.items())
    evidence.update(status="PASS", batch_report=str(batch_path), official_audit=str(audit_path),
                    input_hashes_unchanged=True, official_archive_sha256=official_before,
                    official_dev_scored=False, synthetic_unique_sentences=2, prediction_sentence_records=6,
                    official_dev_counts={k: audit["data"][k] for k in ("dialogues", "sentences", "characters")})
    write_json(directory / "acceptance_summary.json", evidence)
    print("PASS:", directory / "acceptance_summary.json")


if __name__ == "__main__":
    main()
