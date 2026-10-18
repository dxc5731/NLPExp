"""Evidence presence/integrity checks, never a claim of training reproducibility."""
import argparse
import math
from pathlib import Path
import sys

if __package__:
    from .evaluation_io import read_json, resolve_path, sha256, write_json, new_directory, verify_inputs
else:
    from evaluation_io import read_json, resolve_path, sha256, write_json, new_directory, verify_inputs


OFFICIAL_ARCHIVE_SHA256 = "84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67"


def check_run_info(path, evaluation, *, run_id, method_name):
    report = {"schema_version": "1.0", "run_id": run_id, "method_name": method_name,
              "automated_status": "FAIL", "status": "FAIL", "checks": [], "evidence": [],
              "inference_elapsed_seconds_for_comparison": None,
              "notice": "PASS checks file presence, type and consistency only; human claims are not automatically verified."}

    def check(name, ok, message):
        report["checks"].append({"name": name, "status": "PASS" if ok else "FAIL", "message": message})
        return bool(ok)

    def manual(name, applicable=True):
        report["checks"].append({"name": name, "status": "NEEDS_REVIEW" if applicable else "NOT_APPLICABLE",
                                 "message": "Human evidence review required; not proven by this program." if applicable else "Not applicable to this declared run."})

    try:
        path = Path(path).resolve()
        metadata = read_json(path)
        if not isinstance(metadata, dict):
            raise ValueError("run_info must be an object")
        report["run_info"] = {"path": str(path), "sha256": sha256(path)}
    except (OSError, ValueError, TypeError) as exc:
        check("run_info_read", False, str(exc))
        return report
    base = path.parent

    def at(key):
        node = metadata
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        return node

    def text(key):
        value = at(key)
        return check(key, isinstance(value, str) and bool(value.strip()), "Required nonempty string.")

    def file_ref(label, obj, *, hash_required=True):
        try:
            if not isinstance(obj, dict):
                raise ValueError("file reference must be an object")
            target = resolve_path(obj.get("path"), base)
            if not target.is_file():
                raise ValueError("referenced local file not found")
            digest = sha256(target)
            declared = obj.get("sha256")
            if hash_required or declared is not None:
                if declared != digest:
                    raise ValueError("file SHA-256 missing or mismatched")
            report["evidence"].append({"label": label, "path": str(target), "sha256": digest})
            check(label, True, "Local file exists; declared hash checked when required/provided.")
            return True
        except (OSError, ValueError, TypeError) as exc:
            check(label, False, str(exc))
            return False

    synthetic = evaluation.get("purpose") == "synthetic_integration_test"
    check("schema_version", at("schema_version") == "1.0", "Supported metadata schema is 1.0.")
    check("artifact_kind", at("artifact_kind") == ("synthetic_integration_test" if synthetic else "model_prediction"),
          "Synthetic fixtures and actual predictions must not be interchanged.")
    if synthetic:
        text("synthetic_notice")
    check("run_id", at("run_id") == run_id and isinstance(run_id, str) and bool(run_id.strip()), "Must match manifest run_id.")
    check("method.name", at("method.name") == method_name and isinstance(method_name, str) and bool(method_name.strip()), "Must match manifest method name.")
    for key in ("method.kind", "method_version", "data.version", "preprocessing", "config.version"):
        text(key)
    report["method_kind"] = at("method.kind")
    report["method_version"] = at("method_version")
    report["data_version"] = at("data.version")
    check("data.split", at("data.split") == evaluation.get("split"), "Must equal evaluated split.")
    source = evaluation.get("inputs", {}).get("data", {})
    check("data.source_sha256", at("data.source_sha256") == source.get("sha256") and bool(source.get("sha256")), "Must equal scored gold bytes.")
    check("data.archive_sha256", at("data.archive_sha256") == source.get("archive_sha256", None if synthetic else OFFICIAL_ARCHIVE_SHA256),
          "Must match evaluated archive; real raw JSON also declares fixed platform archive provenance.")
    source_file = at("data.source_file")
    if source.get("member"):
        check("data.source_file", source_file == source["member"], "For ZIP inputs use exact member path.")
    else:
        try:
            check("data.source_file", resolve_path(source_file, base) == Path(source["path"]).resolve(), "Must resolve to the scored source JSON.")
        except (ValueError, KeyError, TypeError) as exc:
            check("data.source_file", False, str(exc))
    prediction = evaluation.get("inputs", {}).get("predictions", {})
    check("prediction.sha256", at("prediction.sha256") == prediction.get("sha256") and bool(prediction.get("sha256")), "Must match scored prediction bytes.")
    try:
        check("prediction.file", resolve_path(at("prediction.file"), base) == Path(prediction["path"]).resolve(), "Must identify actual scored file.")
        verify_inputs(evaluation)
        check("input_files_unchanged", True, "Actual source and prediction hashes match evaluation.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        check("input_files_unchanged", False, str(exc))
    revision, dirty = at("code.revision"), at("code.dirty")
    has_revision = isinstance(revision, str) and bool(revision.strip())
    check("code.version", (has_revision and type(dirty) is bool) or (not has_revision and at("code.snapshot") is not None),
          "Provide Git revision+dirty or a local hashed source snapshot.")
    if dirty is True or not has_revision:
        file_ref("code.snapshot", at("code.snapshot"))
    elif at("code.snapshot") is not None:
        file_ref("code.snapshot", at("code.snapshot"))
    if isinstance(at("config.inline"), dict) and at("config.inline"):
        check("config.inline", True, "Complete inline configuration supplied; completeness still needs human review.")
        if at("config.path") is not None or at("config.sha256") is not None:
            file_ref("config.file", at("config"))
    else:
        file_ref("config.file", at("config"))
    logs = at("logs")
    if check("logs", isinstance(logs, list) and bool(logs), "Provide at least one inference/evidence log reference."):
        for index, item in enumerate(logs):
            file_ref(f"logs[{index}]", {"path": item} if isinstance(item, str) else item, hash_required=False)
    alignment = at("alignment")
    check("alignment", (isinstance(alignment, str) and bool(alignment.strip())) or
          (isinstance(alignment, dict) and bool(alignment)), "Describe original-character restoration, or explicitly no tokenizer.")
    kind = str(at("method.kind") or "").lower()
    learned = kind in ("bert", "crf")
    if kind == "bert":
        if isinstance(alignment, dict):
            for key in ("tokenizer", "character_mapping", "special_tokens", "long_text"):
                check("alignment." + key, isinstance(alignment.get(key), str) and bool(alignment[key].strip()), "BERT alignment evidence description required.")
        else:
            manual("alignment_prose_completeness")
    for key in ("inference_command", "inference_working_directory", "timing_scope"):
        if synthetic and at(key) is None:
            report["checks"].append({"name": key, "status": "NOT_APPLICABLE", "message": "Explicit synthetic fixture; no inference occurred."})
        else:
            text(key)
    device = at("device")
    check("device", (synthetic and device is None) or (isinstance(device, str) and bool(device.strip())) or
          (isinstance(device, dict) and bool(device)), "Record actual device; synthetic no-inference fixture may use null.")
    env = at("environment")
    check("environment", isinstance(env, dict) and isinstance(env.get("python"), str) and bool(env["python"]) and
          isinstance(env.get("packages"), dict) and all(isinstance(k, str) and bool(k.strip()) and
          isinstance(v, str) and bool(v.strip()) for k, v in env["packages"].items()),
          "Provide Python version and string package versions (empty packages allowed for stdlib rules).")
    elapsed = at("inference_elapsed_seconds")
    valid_time = type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0
    check("inference_elapsed_seconds", (synthetic and elapsed is None) or valid_time, "Measured finite nonnegative seconds; synthetic not-executed uses null.")
    training = at("training")
    if synthetic:
        report["checks"].append({"name": "training_and_seeds", "status": "NOT_APPLICABLE", "message": "Synthetic integration fixture, no model training."})
    elif training is not None:
        check("training", isinstance(training, dict), "Declare training/dictionary construction provenance.")
        if isinstance(training, dict):
            for key in ("command", "model_version"):
                check("training." + key, isinstance(training.get(key), str) and bool(training[key].strip()), "Required training/construction record.")
            sources = training.get("data")
            if check("training.data", isinstance(sources, list) and bool(sources), "Declare hashed allowed training sources."):
                for i, item in enumerate(sources):
                    check(f"training.data[{i}].split", isinstance(item, dict) and item.get("split") == "train", "Declared build/training split must be train; does not prove no leakage.")
                    file_ref(f"training.data[{i}]", item)
            train_logs = training.get("logs")
            if check("training.logs", isinstance(train_logs, list) and bool(train_logs), "Training/build logs required."):
                for i, item in enumerate(train_logs):
                    file_ref(f"training.logs[{i}]", {"path": item} if isinstance(item, str) else item, hash_required=False)
            file_ref("training.config", training.get("config"))
        seeds = at("random_seeds")
        check("random_seeds", (isinstance(seeds, dict) and bool(seeds) and all(type(v) is int for v in seeds.values())) or
              (not learned and seeds is None and isinstance(at("seed_applicability"), str) and bool(at("seed_applicability"))),
              "Learned/stochastic methods require actual seeds; deterministic construction may declare not applicable.")
    else:
        check("training_applicability", not learned and isinstance(at("training_applicability"), str) and bool(at("training_applicability")), "Learned methods need training evidence; rules explicitly declare no training/build or supply construction records.")
        check("seed_applicability", at("random_seeds") is None and isinstance(at("seed_applicability"), str) and bool(at("seed_applicability")), "Explicit deterministic/no-randomness declaration required.")
    for name, applicable in (("actual_model_identity", not synthetic), ("absence_of_data_leakage", not synthetic),
                             ("faithful_character_alignment", not synthetic), ("actual_training_reproducibility", not synthetic and training is not None),
                             ("truth_of_inference_timing", not synthetic)):
        manual(name, applicable)
    review = at("manual_review")
    if not synthetic and isinstance(review, dict) and review.get("timing_verified") is True:
        has_reviewer = isinstance(review.get("reviewer"), str) and bool(review["reviewer"].strip())
        check("manual_review.reviewer", has_reviewer, "A human reviewer must be identified.")
        if file_ref("manual_review.record", review.get("record")) and has_reviewer and valid_time:
            report["inference_elapsed_seconds_for_comparison"] = elapsed
            report["timing_display_basis"] = "human_attestation_with_hashed_local_review_record; not automatically proven"
    report["automated_status"] = "FAIL" if any(c["status"] == "FAIL" for c in report["checks"]) else "PASS"
    report["status"] = "FAIL" if report["automated_status"] == "FAIL" else (
        "NEEDS_REVIEW" if any(c["status"] == "NEEDS_REVIEW" for c in report["checks"]) else "PASS")
    if report["automated_status"] != "PASS":
        report["inference_elapsed_seconds_for_comparison"] = None
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--run-info", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    evaluation = read_json(args.evaluation)
    report = check_run_info(args.run_info, evaluation, run_id=args.run_id, method_name=args.method)
    directory = new_directory(args.output_dir, input_files=[args.evaluation, args.run_info,
        evaluation["inputs"]["data"]["path"], evaluation["inputs"]["predictions"]["path"]])
    write_json(directory / "reproducibility_check.json", report)
    print(report["status"], directory)
    return 0 if report["automated_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
