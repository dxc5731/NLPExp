"""Compare existing validated metrics, with explicit exclusions and no causal claims."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

if __package__:
    from .check_run_info import check_run_info, OFFICIAL_ARCHIVE_SHA256
    from .evaluation_io import read_json, sha256, verify_inputs, verify_metrics, write_json, write_csv, new_directory, csv_cell
else:
    from check_run_info import check_run_info, OFFICIAL_ARCHIVE_SHA256
    from evaluation_io import read_json, sha256, verify_inputs, verify_metrics, write_json, write_csv, new_directory, csv_cell


OVERVIEW_FIELDS = ("method_name", "run_id", "method_version", "split", "data_version", "gold_sha256",
                   "entity_micro_precision", "entity_micro_recall", "entity_micro_f1", "character_accuracy",
                   "character_micro_f1_excluding_o", "sentences", "inference_elapsed_seconds",
                   "metadata_status", "automated_metadata_status", "purpose", "evaluation_sha256", "prediction_sha256")
TYPE_FIELDS = ("method_name", "run_id", "entity_type", "precision", "recall", "f1", "tp", "fp", "fn", "support")
OFFICIAL_DEV_SHA256 = "16f597ad368971c7db994ca904e05d91942f0e3c73009c9ef1d597fde065d39f"


def comparison_key(result, metadata):
    return (result["split"], result["inputs"]["data"]["sha256"], metadata["data_version"],
            json.dumps(result["metric_policy"], sort_keys=True), tuple(result["entities"]["per_type"]),
            tuple(result["characters"]["confusion_matrix"]["labels"]))


def compare_evaluations(paths, *, purpose):
    if purpose not in ("model_evaluation", "synthetic_integration_test"):
        raise ValueError("Only actual or explicitly synthetic batch comparisons are supported")
    report = {"schema_version": "1.0", "purpose": purpose, "status": "NO_COMPARABLE_RESULTS",
              "scope": "metrics-compatible and automated metadata-complete; not proof of fair training or actual reproducibility",
              "included": [], "excluded": [], "overview": [], "entity_types": []}
    eligible = []
    for value in paths:
        path = Path(value).resolve()
        entry = {"evaluation_path": str(path)}
        try:
            result = read_json(path)
            entry["evaluation_sha256"] = sha256(path)
            entry.update(run_id=result.get("run", {}).get("run_id"), method_name=result.get("run", {}).get("method_name"))
            verify_metrics(result)
            if result.get("purpose") != purpose:
                raise ValueError("PURPOSE_MISMATCH: artificial results cannot enter actual comparison")
            if purpose == "model_evaluation":
                source = result["inputs"]["data"]
                if result.get("split") != "dev" or source["sha256"] != OFFICIAL_DEV_SHA256 or (
                        source.get("member") and source.get("archive_sha256") != OFFICIAL_ARCHIVE_SHA256):
                    raise ValueError("OFFICIAL_DATA_MISMATCH: actual comparison requires fixed platform dev bytes")
            identity = result.get("run", {})
            if not identity.get("run_id") or not identity.get("method_name") or not identity.get("run_info_path"):
                raise ValueError("MISSING_IDENTITY_OR_METADATA: adopt this run through the batch entry point")
            if sha256(identity["run_info_path"]) != identity.get("run_info_sha256"):
                raise ValueError("run_info hash differs from archived evaluation")
            verify_inputs(result)
            metadata = check_run_info(identity["run_info_path"], result,
                                      run_id=identity["run_id"], method_name=identity["method_name"])
            entry["metadata_status"] = metadata["status"]
            if metadata["automated_status"] != "PASS":
                raise ValueError("METADATA_INCOMPLETE_OR_INCONSISTENT: " + ", ".join(c["name"] for c in metadata["checks"] if c["status"] == "FAIL"))
            if metadata["method_version"] != identity.get("method_version") or metadata["data_version"] != identity.get("data_version"):
                raise ValueError("archived identity differs from run_info")
            policy = result["metric_policy"]
            if policy.get("version") != "strict-ner-character-v1" or policy.get("zero_division") != 0 or policy.get("primary_metric") != "entities.micro.f1":
                raise ValueError("unsupported scoring protocol")
            eligible.append((entry, result, metadata))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            entry["reason"] = str(exc)
            report["excluded"].append(entry)
    duplicates = {run_id for run_id, n in Counter(e[0]["run_id"] for e in eligible).items() if n > 1}
    if duplicates:
        valid = []
        for entry, result, metadata in eligible:
            if entry["run_id"] in duplicates:
                report["excluded"].append({**entry, "reason": "DUPLICATE_RUN_ID"})
            else:
                valid.append((entry, result, metadata))
        eligible = valid
    keys = {comparison_key(r, m) for _, r, m in eligible}
    if len(keys) > 1:
        report["status"] = "INCOMPATIBLE"
        report["excluded"].extend({**e, "reason": "INCOMPATIBLE_GOLD_SPLIT_DATA_VERSION_OR_METRIC_POLICY"} for e, _, _ in eligible)
        return report
    eligible.sort(key=lambda item: (-item[1]["entities"]["micro"]["f1"], item[0]["run_id"]))
    for entry, result, metadata in eligible:
        identity = result["run"]
        micro = result["entities"]["micro"]
        report["included"].append({**entry, "run": identity, "inputs": result["inputs"],
                                   "entities": result["entities"], "characters": result["characters"],
                                   "metadata_status": metadata["status"], "automated_metadata_status": metadata["automated_status"],
                                   "human_review_pending": [c["name"] for c in metadata["checks"] if c["status"] == "NEEDS_REVIEW"]})
        report["overview"].append({"method_name": identity["method_name"], "run_id": identity["run_id"],
            "method_version": metadata["method_version"], "split": result["split"], "data_version": metadata["data_version"],
            "gold_sha256": result["inputs"]["data"]["sha256"], "entity_micro_precision": micro["precision"],
            "entity_micro_recall": micro["recall"], "entity_micro_f1": micro["f1"],
            "character_accuracy": result["characters"]["accuracy"],
            "character_micro_f1_excluding_o": result["characters"]["excluding_o"]["micro"]["f1"],
            "sentences": result["counts"]["sentences"],
            "inference_elapsed_seconds": metadata["inference_elapsed_seconds_for_comparison"],
            "metadata_status": metadata["status"], "automated_metadata_status": metadata["automated_status"],
            "purpose": purpose, "evaluation_sha256": entry["evaluation_sha256"],
            "prediction_sha256": result["inputs"]["predictions"]["sha256"]})
        for kind, metric in result["entities"]["per_type"].items():
            report["entity_types"].append({"method_name": identity["method_name"], "run_id": identity["run_id"],
                                           "entity_type": kind, **{k: metric[k] for k in TYPE_FIELDS[3:]}})
    if eligible:
        report["status"] = "COMPARED"
        report["metric_policy"] = eligible[0][1]["metric_policy"]
    return report


def save_comparison(report, directory, *, bom=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "comparison.json", report)
    write_csv(directory / "model_comparison.csv", OVERVIEW_FIELDS, report["overview"], bom=bom)
    write_csv(directory / "entity_type_comparison.csv", TYPE_FIELDS, report["entity_types"], bom=bom)
    verify_comparison_files(report, directory)


def verify_comparison_files(report, directory):
    """Reopen JSON/CSVs and check every cell against the source evaluations."""
    directory = Path(directory)
    if read_json(directory / "comparison.json") != report:
        raise ValueError("saved comparison JSON differs")
    for name, fields, key in (("model_comparison.csv", OVERVIEW_FIELDS, "overview"),
                              ("entity_type_comparison.csv", TYPE_FIELDS, "entity_types")):
        with (directory / name).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(fields):
                raise ValueError("comparison CSV columns differ")
            actual = list(reader)
        expected = [{k: "" if row.get(k) is None else str(csv_cell(row.get(k, ""))) for k in fields}
                    for row in report[key]]
        if actual != expected:
            raise ValueError("saved comparison CSV cells differ from summary")
    for entry in report["included"]:
        result = read_json(entry["evaluation_path"])
        if (sha256(entry["evaluation_path"]) != entry["evaluation_sha256"] or
                result["entities"] != entry["entities"] or result["characters"] != entry["characters"]):
            raise ValueError("comparison differs from independent evaluation")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--purpose", required=True, choices=("model_evaluation", "synthetic_integration_test"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--csv-bom", action="store_true")
    args = parser.parse_args()
    report = compare_evaluations(args.evaluations, purpose=args.purpose)
    inputs = list(args.evaluations)
    for item in report["included"]:
        inputs.extend([item["inputs"]["data"]["path"], item["inputs"]["predictions"]["path"], item["run"]["run_info_path"]])
    root = new_directory(args.output_dir, input_files=inputs)
    save_comparison(report, root / "comparison", bom=args.csv_bom)
    for row in report["overview"]:
        print(f"{row['run_id']}: entity micro-F1={row['entity_micro_f1']:.6f} metadata={row['metadata_status']}")
    print(report["status"], "excluded=", len(report["excluded"]), root)
    return 0 if report["status"] == "COMPARED" and not report["excluded"] else 1


if __name__ == "__main__":
    sys.exit(main())
