"""Validate first, then score strict entities and character classes on CPU.

Standard library only. Each CLI attempt gets a new directory; never overwrites.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter

if __package__:
    from . import ner_metrics, validate_predictions
else:
    import ner_metrics
    import validate_predictions


PURPOSES = ("scorer_validation", "model_evaluation", "synthetic_integration_test")


def evaluate(data_path, prediction_path, *, split="dev", purpose, max_examples=30, return_inputs=False):
    """Return a traceable result or rejection WITHOUT any metric keys.

    Input parsing/validation happens once. Scoring uses exactly the in-memory
    objects whose original bytes were hashed by the validator.
    """
    if purpose not in PURPOSES:
        raise ValueError("purpose must explicitly distinguish validation from model evaluation")
    if split not in ("dev", "test"):
        raise ValueError("this entry point supports dev scoring and explicit test rejection only")
    started = perf_counter()
    validation, data, predictions = validate_predictions.load_and_validate(
        data_path, prediction_path, split=split, max_examples=max_examples)
    after_validation = perf_counter()
    result = {
        "schema_version": "1.0", "purpose": purpose, "split": split,
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "python_version": platform.python_version(), "status": "REJECTED",
        "inputs": validation["inputs"], "validation": validation,
        "timing_seconds": {"validation": after_validation - started, "scoring": None},
        "implementation": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(ner_metrics.__file__), Path(validate_predictions.__file__))},
    }
    if validation["status"] != "PASS":
        result["rejection"] = {"code": "VALIDATION_" + validation["status"],
                               "message": "Scoring not run: full input validation must PASS."}
    elif split == "test":
        result["rejection"] = {"code": "TEST_HAS_NO_PUBLIC_GOLD",
                               "message": "Official test has no public BIO gold; local scoring is refused."}
    else:
        gold, predicted = {}, {}
        for did, dialogue in data.items():
            for row in dialogue["dialogue"]:
                sid = row["sentence_id"]
                key = (did, sid)
                gold[key] = row["BIO_label"].split(" ") if row["BIO_label"] else []
                value = predictions[did][sid]
                predicted[key] = value.split(" ") if value else []
        score_start = perf_counter()
        metrics = ner_metrics.score_sequences(gold, predicted)
        result["timing_seconds"]["scoring"] = perf_counter() - score_start
        result.update(status="SCORED", counts={k: validation["data"][k] for k in (
            "dialogues", "sentences", "characters")}, **metrics)
        result["metric_policy"] = {
            "version": "strict-ner-character-v1", "scale": "fractions in [0,1], unrounded",
            "primary_metric": "entities.micro.f1", "zero_division": 0.0,
            "entity_identity": ["dialogue_id", "sentence_id", "entity_type", "start", "end"],
            "span": "zero-based [start, end), Unicode code points",
            "entity_matching": "exact type and boundaries within original dialogue and sentence",
            "entity_micro": "pool TP/FP/FN globally before computing P/R/F1",
            "bio_policy": "strict, reject invalid BIO without repair",
            "character_unit": "Python len(str), original Unicode code points, no normalization",
            "character_accuracy": "all positions including O; correct / total",
            "character_micro": "sum one-vs-rest TP/FP/FN over selected labels, using all positions",
            "character_macro": "unweighted mean over all fixed 11 labels (with O) or 10 (without O); absent F1=0",
            "excluding_o": "exclude O label only, retain O->entity FP and entity->O FN; no position filtering",
            "confusion_direction": "rows=gold, columns=predicted; fixed labels in characters.confusion_matrix.labels",
            "support": "gold count for the entity type or character label",
        }
    result["timing_seconds"]["total_before_publication"] = perf_counter() - started
    return (result, data, predictions) if return_inputs else result


def save_result(result, output_dir, *, data_path, prediction_path):
    """Exclusive publication into a fresh child directory, including rejections."""
    root = Path(output_dir).resolve()
    if validate_predictions.protected_report_path(root, data_path, prediction_path):
        raise ValueError("output directory cannot be an input or inside the supplied data directory")
    root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S%f") + "_"
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    filename = "evaluation.json" if result["status"] == "SCORED" else "rejection.json"
    path = run_dir / filename
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Raw dev JSON, data directory, or original ZIP")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--purpose", required=True, choices=PURPOSES,
                        help="scorer_validation for artificial fixtures; model_evaluation for actual predictions")
    parser.add_argument("--output-dir", type=Path, help="Parent for a fresh unique attempt directory")
    parser.add_argument("--max-examples", default=30, type=int)
    args = parser.parse_args(argv)
    if args.max_examples < 0:
        parser.error("--max-examples must be nonnegative")
    output_dir = args.output_dir or Path("logs/evaluation" if args.purpose == "scorer_validation" else "results/evaluation")
    if validate_predictions.protected_report_path(output_dir, args.data, args.predictions):
        parser.error("--output-dir cannot be an input or inside the supplied data directory")
    result = evaluate(args.data, args.predictions, split=args.split,
                      purpose=args.purpose, max_examples=args.max_examples)
    try:
        report_path = save_result(result, output_dir, data_path=args.data, prediction_path=args.predictions)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot publish report ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    print(f"{result['status']} purpose={args.purpose} split={args.split}")
    if result["status"] != "SCORED":
        print(result["rejection"]["code"] + ": " + result["rejection"]["message"])
        print("Validation errors: " + json.dumps(result["validation"]["errors"]["by_code"]))
        print(f"Rejection report: {report_path}")
        return 1 if result["validation"]["status"] == "FAIL" else 2
    entities, chars = result["entities"], result["characters"]
    for name, metric in [("ENTITY MICRO (primary)", entities["micro"]), *entities["per_type"].items()]:
        print(f"{name}: TP={metric['tp']} FP={metric['fp']} FN={metric['fn']} "
              f"P={metric['precision']:.6f} R={metric['recall']:.6f} F1={metric['f1']:.6f}")
    print(f"Gold entities={entities['gold_total']} Predicted entities={entities['predicted_total']}")
    print(f"Character accuracy (with O)={chars['accuracy']:.6f} "
          f"correct={chars['correct']} total={chars['total']}")
    for name in ("including_o", "excluding_o"):
        summary = chars[name]
        m = summary["micro"]
        print(f"Character {name}: micro-P={m['precision']:.6f} micro-R={m['recall']:.6f} "
              f"micro-F1={m['f1']:.6f} macro-F1={summary['macro_f1']:.6f}")
    print(f"Result: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
