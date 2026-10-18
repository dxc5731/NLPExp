"""Runnable artificial-fixture entity crosscheck; creates evidence, not scores for a model."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from author_entity_reference import author_entity_reference
from src.evaluate_predictions import evaluate


def crosscheck():
    base = ROOT / "examples" / "scorer_validation"
    gold = json.loads((base / "data_dev.json").read_text(encoding="utf-8"))
    prediction = json.loads((base / "predictions_dev.json").read_text(encoding="utf-8"))
    rounded_gold = {"d": {"dialogue": [{"sentence_id": "1", "sentence": "甲乙丙丁戊己",
        "BIO_label": "B-Symptom B-Symptom B-Drug B-Drug_Category B-Medical_Examination B-Operation"}]}}
    rounded_prediction = {"d": {"1": "B-Symptom O B-Drug B-Drug_Category B-Medical_Examination B-Operation"}}
    results = []
    for name, case_gold, pred, expected_f1 in (
        ("mixed_five_types", gold, prediction, 4 / 9),
        ("identity_diagnostic", gold, {did: {row["sentence_id"]: row["BIO_label"] for row in d["dialogue"]}
                                  for did, d in gold.items()}, 1.0),
        ("all_o_diagnostic", gold, {did: {row["sentence_id"]: " ".join(["O"] * len(row["sentence"]))
                                  for row in d["dialogue"]} for did, d in gold.items()}, 0.0),
        ("class_rounding_diagnostic", rounded_gold, rounded_prediction, 10 / 11),
    ):
        # Pure API for derived synthetic cases; mixed fixture additionally runs
        # the complete validated file evaluator on the exact fixture bytes.
        from src.ner_metrics import score_sequences
        g = {(did, row["sentence_id"]): row["BIO_label"].split(" ")
             for did, d in case_gold.items() for row in d["dialogue"]}
        p = {(did, sid): value.split(" ") for did, rows in pred.items() for sid, value in rows.items()}
        ours = score_sequences(g, p)["entities"]
        if name == "mixed_five_types":
            full = evaluate(base / "data_dev.json", base / "predictions_dev.json",
                            purpose="scorer_validation")
            if full["status"] != "SCORED" or full["entities"] != ours:
                raise AssertionError("validated CLI core differs from pure metric results")
        author = author_entity_reference(case_gold, pred)
        if abs(ours["micro"]["f1"] - expected_f1) > 1e-12:
            raise AssertionError("manual overall expectation mismatch")
        if abs(ours["micro"]["f1"] - author["overall_f1"]) > 1e-12:
            raise AssertionError("author overall mismatch")
        classes = {}
        for kind, metric in ours["per_type"].items():
            ref = author["class_info"][kind]["f1"]
            delta = abs(metric["f1"] - ref)
            if delta > 0.000050000001:
                raise AssertionError("author rounded class F1 mismatch")
            classes[kind] = {"ours": metric["f1"], "author_rounded_4dp": ref, "absolute_difference": delta}
        results.append({"case": name, "expected_overall_f1": expected_f1, "ours": ours,
                        "author": author, "per_type_comparison": classes, "status": "PASS"})
    return {"purpose": "scorer_validation", "status": "PASS",
            "notice": "Entirely artificial data; no model experiment, no Tianchi server, no seqeval/character equivalence claim.",
            "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
            "overall_tolerance": 1e-12, "rounded_class_tolerance": 0.000050000001, "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    result = crosscheck()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"PASS: {len(result['cases'])} artificial cases, overall and all five types; {args.report.resolve()}")


if __name__ == "__main__":
    main()
