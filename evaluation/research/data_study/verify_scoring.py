"""Small synthetic crosscheck against the pinned author's entity code.

No official dataset, model predictions, seqeval or model runtime is loaded.
The all-types anchor is explicit fixture scaffolding, not a model result.
"""

import argparse
import ast
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from author_entity_reference import (  # noqa: E402
    AUTHOR_PATH, AUTHOR_SHA256, AUTHOR_URL, author_entity_reference,
)
from src.ner_metrics import score_sequences  # noqa: E402
from src.validate_predictions import ENTITY_TYPES  # noqa: E402


TOLERANCE = 1e-12
ROUNDING_TOLERANCE = 0.000050000001
ANCHOR = tuple("B-" + kind for kind in ENTITY_TYPES)
S = "Symptom"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixtures():
    """Eight manually counted cases; every legal case includes five correct anchors."""
    # Expected tuples are (TP, FP, FN); fractions are written by hand for pooled P/R/F1.
    return [
        dict(name="exact_match", targets=[("d", "1", "B-Symptom I-Symptom O B-Drug",
                                           "B-Symptom I-Symptom O B-Drug")],
             counts=(7, 0, 0), ratios=("1", "1", "1"),
             overrides={S: (2, 0, 0), "Drug": (2, 0, 0)}),
        dict(name="shifted_start", targets=[("d", "1", "O B-Symptom I-Symptom O",
                                             "B-Symptom I-Symptom I-Symptom O")],
             counts=(5, 1, 1), ratios=("5/6", "5/6", "5/6"),
             overrides={S: (1, 1, 1)}),
        dict(name="wrong_type", targets=[("d", "1", "B-Symptom I-Symptom",
                                         "B-Drug I-Drug")],
             counts=(5, 1, 1), ratios=("5/6", "5/6", "5/6"),
             overrides={S: (1, 0, 1), "Drug": (1, 1, 0)}),
        dict(name="merge_adjacent_entities", targets=[("d", "1", "B-Symptom I-Symptom B-Symptom I-Symptom",
                                                       "B-Symptom I-Symptom I-Symptom I-Symptom")],
             counts=(5, 1, 2), ratios=("5/6", "5/7", "10/13"),
             overrides={S: (1, 1, 2)}),
        dict(name="split_one_entity", targets=[("d", "1", "B-Symptom I-Symptom I-Symptom I-Symptom",
                                                "B-Symptom I-Symptom B-Symptom I-Symptom")],
             counts=(5, 2, 1), ratios=("5/7", "5/6", "10/13"),
             overrides={S: (1, 2, 1)}),
        dict(name="adjacent_same_type_kept_separate", targets=[("d", "1", "B-Symptom B-Symptom",
                                                               "B-Symptom B-Symptom")],
             counts=(7, 0, 0), ratios=("1", "1", "1"),
             overrides={S: (3, 0, 0)}),
        dict(name="sentence_identity_isolation", targets=[("d", "1", "B-Symptom I-Symptom", "O O"),
                                                          ("d", "2", "O O", "B-Symptom I-Symptom")],
             counts=(5, 1, 1), ratios=("5/6", "5/6", "5/6"),
             overrides={S: (1, 1, 1)}),
        dict(name="dialogue_identity_isolation", targets=[("d1", "1", "B-Symptom I-Symptom", "O O"),
                                                          ("d2", "1", "O O", "B-Symptom I-Symptom")],
             counts=(5, 1, 1), ratios=("5/6", "5/6", "5/6"),
             overrides={S: (1, 1, 1)}),
    ]


def to_author_inputs(gold, predicted):
    data, prediction = {}, {}
    for (dialogue_id, sentence_id), labels in gold.items():
        # Artificial placeholder text only; the author entity code reads labels, not text.
        data.setdefault(dialogue_id, {"dialogue": []})["dialogue"].append(
            {"sentence_id": sentence_id, "sentence": "甲" * len(labels), "BIO_label": " ".join(labels)})
        prediction.setdefault(dialogue_id, {})[sentence_id] = " ".join(predicted[(dialogue_id, sentence_id)])
    return data, prediction


def close(actual, expected, tolerance=TOLERANCE):
    if abs(actual - float(expected)) > tolerance:
        raise AssertionError(f"expected {expected}, got {actual}")


def check_case(case):
    gold = {("anchor", "five_types"): ANCHOR}
    predicted = dict(gold)
    for did, sid, g, p in case["targets"]:
        gold[(did, sid)], predicted[(did, sid)] = g.split(" "), p.split(" ")
    ours = score_sequences(gold, predicted)["entities"]
    author = author_entity_reference(*to_author_inputs(gold, predicted))
    tp, fp, fn = case["counts"]
    if tuple(ours["micro"][key] for key in ("tp", "fp", "fn")) != (tp, fp, fn):
        raise AssertionError("ours differs from manually counted pooled entities")
    author_counts = (author["tp"], author["predicted_total"] - author["tp"],
                     author["gold_total"] - author["tp"])
    if author_counts != (tp, fp, fn):
        raise AssertionError("author differs from manually counted pooled entities")
    _, author_compute = original_helpers()
    author_recall, author_precision, author_f1 = author_compute(
        author["gold_total"], author["predicted_total"], author["tp"])
    author_micro = dict(zip(("tp", "fp", "fn"), author_counts))
    author_micro.update(precision=author_precision, recall=author_recall, f1=author_f1)
    for metric, expected in zip(("precision", "recall", "f1"), case["ratios"]):
        close(ours["micro"][metric], Fraction(expected))
        close(author_micro[metric], Fraction(expected))
    close(author["overall_f1"], Fraction(case["ratios"][2]))
    per_type = {}
    for kind in ENTITY_TYPES:
        expected_counts = case["overrides"].get(kind, (1, 0, 0))
        actual = ours["per_type"][kind]
        if tuple(actual[key] for key in ("tp", "fp", "fn")) != expected_counts:
            raise AssertionError(f"{kind}: manual per-type counts differ")
        # Exact rational arithmetic from the explicitly hand-listed counts.
        t, f, n = expected_counts
        expected_prf = (Fraction(t, t + f), Fraction(t, t + n), Fraction(2 * t, 2 * t + f + n))
        for metric, expected in zip(("precision", "recall", "f1"), expected_prf):
            close(actual[metric], expected)
        reference = author["class_info"][kind]
        for ours_key, author_key in (("precision", "acc"), ("recall", "recall"), ("f1", "f1")):
            close(actual[ours_key], reference[author_key], ROUNDING_TOLERANCE)
        per_type[kind] = {"manual_counts": dict(zip(("tp", "fp", "fn"), expected_counts)),
                          "ours": actual, "author_rounded": reference}
    return {"name": case["name"], "status": "PASS", "manual_counts": dict(zip(("tp", "fp", "fn"), case["counts"])),
            "manual_prf_fractions": dict(zip(("precision", "recall", "f1"), case["ratios"])),
            "ours_micro": ours["micro"], "author_counts": dict(zip(("tp", "fp", "fn"), author_counts)),
            "author_micro": author_micro, "author_overall_f1": author["overall_f1"], "per_type": per_type,
            "synthetic_sequences": [{"dialogue_id": k[0], "sentence_id": k[1],
                                      "gold": list(gold[k]), "prediction": list(predicted[k])} for k in gold]}


def original_helpers():
    """Load only two unchanged author functions, without importing seqeval."""
    if sha256(AUTHOR_PATH) != AUTHOR_SHA256:
        raise ValueError("pinned author source hash mismatch")
    tree = ast.parse(AUTHOR_PATH.read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("get_entity_bio", "compute")]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 str(AUTHOR_PATH) + ":original_helpers_only", "exec"), namespace)
    return namespace["get_entity_bio"], namespace["compute"]


def protocol_checks():
    extract, compute = original_helpers()
    gold = {("d", "1"): ["O", "O"]}
    ours = score_sequences(gold, gold)["entities"]["micro"]
    if any(ours[k] != 0 for k in ("tp", "fp", "fn", "precision", "recall", "f1")):
        raise AssertionError("our zero-denominator policy differs")
    if extract(["O", "O"]) != [] or compute(0, 0, 0) != (0, 0, 0.0):
        raise AssertionError("author helper zero policy differs")
    try:
        author_entity_reference(*to_author_inputs(gold, gold))
    except KeyError as exc:
        exception = {"type": type(exc).__name__, "missing_key": str(exc.args[0])}
    else:
        raise AssertionError("expected pinned full entity prefix to reject absent gold classes")
    rows = []
    for labels, expected in [(["I-Symptom"], []),
                             (["B-Symptom", "I-Drug"], [["Symptom", 0, 0]])]:
        try:
            score_sequences({("d", "1"): ["O"] * len(labels)}, {("d", "1"): labels})
        except ValueError:
            pass
        else:
            raise AssertionError("our strict BIO policy must reject illegal I transitions")
        author_entities = extract(labels)
        if author_entities != expected:
            raise AssertionError("unexpected pinned author's illegal BIO extraction behavior")
        rows.append({"prediction": labels, "ours": "REJECTED", "author_helper_entities": author_entities})
    return [
        {"name": "no_entities_zero_and_missing_classes", "status": "EXPECTED_PROTOCOL_DIFFERENCE",
         "ours": ours, "author_original_helpers_zero_prf": list(compute(0, 0, 0)),
         "author_full_entity_prefix": exception,
         "meaning": "Helper formulas agree at zero; the unmodified full author entity prefix assumes all five gold classes exist."},
        {"name": "illegal_bio_policy", "status": "EXPECTED_PROTOCOL_DIFFERENCE", "cases": rows,
         "meaning": "Invalid BIO is refused before our scoring; permissive author helper output is not a valid score comparison."},
    ]


def run_checks():
    started = time.perf_counter()
    cases = [check_case(case) for case in fixtures()]
    protocols = protocol_checks()
    return {"schema_version": "1.0", "purpose": "scorer_validation", "status": "PASS",
            "notice": "Entirely synthetic verification; not model results, not a full author CLI/seqeval/Tianchi server run.",
            "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
            "anchor": {"gold_entities": 5, "predicted_entities": 5, "tp": 5,
                       "reason": "Make all five gold classes present for legal author comparisons; manual counts explicitly include this anchor."},
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "sources": {"author_url": AUTHOR_URL, "author_path": str(AUTHOR_PATH), "author_sha256": sha256(AUTHOR_PATH),
                        "reference_adapter_sha256": sha256(ROOT / "tests" / "author_entity_reference.py"),
                        "our_metrics_sha256": sha256(ROOT / "src" / "ner_metrics.py"),
                        "validator_sha256": sha256(ROOT / "src" / "validate_predictions.py"),
                        "verification_script_sha256": sha256(__file__)},
            "tolerances": {"overall": TOLERANCE, "author_per_type_rounded": ROUNDING_TOLERANCE},
            "legal_cases": cases, "legal_case_count": len(cases), "protocol_checks": protocols,
            "protocol_check_count": len(protocols), "elapsed_seconds": time.perf_counter() - started,
            "remaining_limits": ["No real prediction or official dev gold was read.",
                                 "Character metrics were not independently reimplemented or crosschecked here.",
                                 "No third-party seqeval execution, full author CLI or Tianchi server verification.",
                                 "Correct scoring does not prove honest training, no leakage, or faithful tokenizer alignment."]}


def save_report(report, output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S%f_")
    destination = Path(tempfile.mkdtemp(prefix=prefix, dir=output_dir)) / "scoring_crosscheck.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "reliability_runs")
    args = parser.parse_args()
    report = run_checks()
    report["command"] = [sys.executable, *sys.argv]
    path = save_report(report, args.output_dir)
    print(f"PASS: {report['legal_case_count']} synthetic legal cases; "
          f"{report['protocol_check_count']} expected protocol differences; {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
