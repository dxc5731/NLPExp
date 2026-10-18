"""One command: validate -> score -> errors -> evidence checks -> comparison."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from time import perf_counter

if __package__:
    from .evaluate_predictions import evaluate
    from .check_run_info import check_run_info
    from .export_errors import export_errors, verify_error_files
    from .compare_evaluations import compare_evaluations, save_comparison, verify_comparison_files
    from .evaluation_io import read_json, write_json, sha256, resolve_path, new_directory, verify_inputs, verify_metrics
else:
    from evaluate_predictions import evaluate
    from check_run_info import check_run_info
    from export_errors import export_errors, verify_error_files
    from compare_evaluations import compare_evaluations, save_comparison, verify_comparison_files
    from evaluation_io import read_json, write_json, sha256, resolve_path, new_directory, verify_inputs, verify_metrics


def run_batch(manifest_path, *, output_dir=None, local_text=False, bom=False):
    started = perf_counter()
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0":
        raise ValueError("manifest schema_version must be 1.0")
    purpose = manifest.get("purpose")
    if purpose not in ("model_evaluation", "synthetic_integration_test"):
        raise ValueError("manifest requires actual or explicitly synthetic purpose")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("manifest.runs must be a nonempty array")
    base = manifest_path.parent
    prepared = []
    files, directories = [manifest_path], []
    ids = Counter(r.get("run_id") for r in runs if isinstance(r, dict) and isinstance(r.get("run_id"), str))
    for index, raw in enumerate(runs):
        item = {"index": index + 1, "run_id": raw.get("run_id") if isinstance(raw, dict) else None,
                "method_name": raw.get("method_name") if isinstance(raw, dict) else None}
        try:
            if not isinstance(raw, dict):
                raise ValueError("run entry must be an object")
            if not isinstance(item["run_id"], str) or not item["run_id"].strip() or not isinstance(item["method_name"], str) or not item["method_name"].strip():
                raise ValueError("run_id and method_name must be nonempty strings")
            if ids[item["run_id"]] > 1:
                raise ValueError("DUPLICATE_RUN_ID: all duplicate entries rejected")
            if raw.get("purpose", purpose) != purpose:
                raise ValueError("mixed synthetic/actual purpose is forbidden")
            source = raw.get("data", manifest.get("data"))
            if not isinstance(source, dict) or source.get("split") not in ("dev", "test"):
                raise ValueError("data requires path and split=dev/test")
            item.update(data_path=str(resolve_path(source.get("path"), base)), split=source["split"],
                        prediction_path=str(resolve_path(raw.get("predictions"), base)))
            metadata_path = raw.get("run_info")
            item["run_info_path"] = str(resolve_path(metadata_path, base)) if metadata_path else None
            files.extend([item["prediction_path"]] + ([item["run_info_path"]] if item["run_info_path"] else []))
            data_path = Path(item["data_path"])
            (directories if data_path.is_dir() else files).append(data_path)
        except (ValueError, TypeError) as exc:
            item["precheck_error"] = str(exc)
        prepared.append(item)
    if output_dir is None:
        requested = manifest.get("output_dir")
        if requested:
            output_dir = resolve_path(requested, base)
        else:
            output_dir = Path.cwd() / ("logs/pipeline_validation" if purpose == "synthetic_integration_test" else "results/batches")
    directory = new_directory(output_dir, input_files=files, data_directories=directories)
    report = {"schema_version": "1.0", "purpose": purpose, "status": "FAILURE",
              "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
              "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
              "directory": str(directory), "runs": [], "self_check": "NOT_RUN",
              "notice": "Technical scores are retained even with incomplete metadata. Comparison eligibility is checked separately; no claim of fair training."}
    scored_paths = []
    for item in prepared:
        outcome = dict(item, status="FAILED")
        report["runs"].append(outcome)
        run_dir = directory / "runs" / f"{item['index']:03d}"
        run_dir.mkdir(parents=True)
        try:
            if item.get("precheck_error"):
                raise ValueError(item["precheck_error"])
            result, data, predictions = evaluate(item["data_path"], item["prediction_path"],
                split=item["split"], purpose=purpose, return_inputs=True)
            if result["status"] != "SCORED":
                path = run_dir / "rejection.json"
                write_json(path, result)
                outcome.update(status="REJECTED", reason=result["rejection"], rejection_path=str(path))
                continue
            metadata = check_run_info(item["run_info_path"], result, run_id=item["run_id"], method_name=item["method_name"])
            write_json(run_dir / "reproducibility_check.json", metadata)
            result["run"] = {"run_id": item["run_id"], "method_name": item["method_name"],
                             "method_version": metadata.get("method_version"), "data_version": metadata.get("data_version"),
                             "run_info_path": item["run_info_path"], "run_info_sha256": metadata.get("run_info", {}).get("sha256")}
            result["reproducibility"] = {"status": metadata["status"], "automated_status": metadata["automated_status"],
                                         "report_path": str(run_dir / "reproducibility_check.json")}
            result_path = run_dir / "evaluation.json"
            write_json(result_path, result)
            outcome.update(status="SCORED" if metadata["automated_status"] == "PASS" else "DEBUG_SCORED",
                           evaluation_path=str(result_path), evaluation_sha256=sha256(result_path),
                           metadata_status=metadata["status"], automated_metadata_status=metadata["automated_status"],
                           entity_micro_f1=result["entities"]["micro"]["f1"])
            export_errors(result, data, predictions, run_dir / "errors", run_id=item["run_id"],
                          method_name=item["method_name"], local_text=local_text, bom=bom)
            outcome["error_export"] = "PASS"
            scored_paths.append(result_path)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            outcome.update(status="FAILED", reason=str(exc))
        finally:
            write_json(run_dir / "run_status.json", outcome)
    comparison = compare_evaluations(scored_paths, purpose=purpose)
    save_comparison(comparison, directory / "comparison", bom=bom)
    report["comparison"] = {"status": comparison["status"], "included": len(comparison["included"]),
                            "excluded": comparison["excluded"], "path": str(directory / "comparison/comparison.json")}
    # Final self-check reopens the actual artifacts after all output is produced.
    failures = []
    for outcome in report["runs"]:
        if outcome["status"] in ("SCORED", "DEBUG_SCORED"):
            try:
                result = read_json(outcome["evaluation_path"])
                verify_inputs(result)
                verify_metrics(result)
                verify_error_files(result, Path(outcome["evaluation_path"]).parent / "errors")
                if sha256(outcome["evaluation_path"]) != outcome["evaluation_sha256"]:
                    raise ValueError("evaluation artifact changed")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                outcome.update(status="FAILED", final_self_check_error=str(exc))
                failures.append(outcome)
    comparison_check_error = None
    try:
        verify_comparison_files(comparison, directory / "comparison")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        comparison_check_error = str(exc)
        report["comparison_self_check_error"] = comparison_check_error
    report["self_check"] = "FAIL" if failures or comparison_check_error else "PASS"
    if failures or comparison_check_error:
        # Do not leave a successful-looking comparison when final checks fail.
        comparison["status"] = "INVALIDATED_BY_FINAL_SELF_CHECK"
        comparison["included"], comparison["overview"], comparison["entity_types"] = [], [], []
        for name in ("comparison.json", "model_comparison.csv", "entity_type_comparison.csv"):
            (directory / "comparison" / name).unlink()
        save_dir = directory / "comparison"
        write_json(save_dir / "comparison.json", comparison)
        report["comparison"]["status"] = comparison["status"]
        report["comparison"]["included"] = 0
        for outcome in failures:
            status_path = Path(outcome["evaluation_path"]).parent / "run_status.json"
            status_path.unlink()
            write_json(status_path, outcome)
    scored = sum(r["status"] in ("SCORED", "DEBUG_SCORED") for r in report["runs"])
    if scored:
        report["status"] = "SUCCESS" if all(r["status"] == "SCORED" for r in report["runs"]) and comparison["status"] == "COMPARED" and not comparison["excluded"] and report["self_check"] == "PASS" else "PARTIAL_SUCCESS"
    report["elapsed_seconds"] = perf_counter() - started
    report["implementation"] = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    write_json(directory / "batch_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, help="CLI override is relative to current working directory")
    parser.add_argument("--local-text", action="store_true", help="Export private medical text only under local_analysis/")
    parser.add_argument("--csv-bom", action="store_true", help="UTF-8 BOM for Excel; default UTF-8 without BOM")
    args = parser.parse_args()
    try:
        report = run_batch(args.manifest, output_dir=args.output_dir, local_text=args.local_text, bom=args.csv_bom)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("FAILURE:", str(exc), file=sys.stderr)
        return 1
    for run in report["runs"]:
        print(f"{run['run_id']}: {run['status']} entity_micro_f1={run.get('entity_micro_f1', 'not_scored')} metadata={run.get('metadata_status', 'not_checked')}")
    print(report["status"], "self_check=", report["self_check"], "seconds=", round(report["elapsed_seconds"], 3))
    print("Batch report:", Path(report["directory"]) / "batch_report.json")
    return {"SUCCESS": 0, "PARTIAL_SUCCESS": 2, "FAILURE": 1}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
