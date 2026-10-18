"""Strict entity details and conservative candidate links; no causal attribution."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

if __package__:
    from .ner_metrics import ENTITY_TYPES, extract_entities, score_entity_sets
    from .evaluation_io import read_json, write_json, write_csv, new_directory, verify_inputs, verify_metrics
    from .validate_predictions import load_and_validate
else:
    from ner_metrics import ENTITY_TYPES, extract_entities, score_entity_sets
    from evaluation_io import read_json, write_json, write_csv, new_directory, verify_inputs, verify_metrics
    from validate_predictions import load_and_validate


FIELDS = ("record_id", "run_id", "method_name", "split", "dialogue_id", "sentence_id", "category",
          "entity_type", "start", "end", "character_positions", "coordinate_unit", "candidate_id", "gold_sha256", "prediction_sha256")
CANDIDATE_FIELDS = ("candidate_id", "run_id", "method_name", "split", "dialogue_id", "sentence_id",
                    "phenomenon", "fp_record_id", "fn_record_id", "association_status")


def stable_id(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()[:32]


def entity_key(entity):
    return entity.start, entity.end, entity.entity_type


def candidate_links(fp, fn):
    """Pair only mutually unique choices; ambiguous overlaps remain unpaired."""
    fp, fn = set(fp), set(fn)
    pairs = []
    rules = (
        ("type_mismatch", lambda p, g: p.start == g.start and p.end == g.end and p.entity_type != g.entity_type),
        ("boundary_mismatch", lambda p, g: p.entity_type == g.entity_type and (p.start, p.end) != (g.start, g.end)
         and max(p.start, g.start) < min(p.end, g.end)),
    )
    for phenomenon, matches in rules:
        p_choices = {p: [g for g in fn if matches(p, g)] for p in fp}
        g_choices = {g: [p for p in fp if matches(p, g)] for g in fn}
        chosen = [(p, options[0]) for p, options in p_choices.items()
                  if len(options) == 1 and len(g_choices[options[0]]) == 1]
        for p, g in sorted(chosen, key=lambda pair: (entity_key(pair[0]), entity_key(pair[1]))):
            pairs.append((phenomenon, p, g, "unique_one_to_one"))
            fp.remove(p)
            fn.remove(g)
    for p in sorted(fp, key=entity_key):
        ambiguous = any(rule(p, g) for _, rule in rules for g in fn)
        pairs.append(("suspected_spurious", p, None, "ambiguous_unlinked" if ambiguous else "unlinked"))
    for g in sorted(fn, key=entity_key):
        ambiguous = any(rule(p, g) for _, rule in rules for p in fp)
        pairs.append(("suspected_miss", None, g, "ambiguous_unlinked" if ambiguous else "unlinked"))
    return pairs


def build_details(evaluation, data, predictions, *, run_id, method_name, local_text=False):
    verify_metrics(evaluation)
    errors, matches, candidates, local_rows = [], [], [], []
    gold_all, pred_all = set(), set()
    for did, dialogue in data.items():
        for row in dialogue["dialogue"]:
            sid = row["sentence_id"]
            g = extract_entities(row["BIO_label"].split(" ") if row["BIO_label"] else [], did, sid)
            value = predictions[did][sid]
            p = extract_entities(value.split(" ") if value else [], did, sid)
            gold_all.update(g)
            pred_all.update(p)
            fp, fn, tp = p - g, g - p, p & g
            records = {}
            for category, entities in (("TP", tp), ("FP", fp), ("FN", fn)):
                for entity in sorted(entities, key=entity_key):
                    record = {"record_id": stable_id(run_id, did, sid, category, *entity_key(entity)),
                              "run_id": run_id, "method_name": method_name, "split": evaluation["split"],
                              "dialogue_id": did, "sentence_id": sid, "category": category,
                              "entity_type": entity.entity_type, "start": entity.start, "end": entity.end,
                              "character_positions": json.dumps(list(range(entity.start, entity.end))),
                              "coordinate_unit": "original Unicode code points, zero-based [start,end)",
                              "candidate_id": "", "gold_sha256": evaluation["inputs"]["data"]["sha256"],
                              "prediction_sha256": evaluation["inputs"]["predictions"]["sha256"]}
                    records[(category, entity)] = record
                    (matches if category == "TP" else errors).append(record)
            for phenomenon, pre, gold, association in candidate_links(fp, fn):
                p_id = records[("FP", pre)]["record_id"] if pre is not None else ""
                g_id = records[("FN", gold)]["record_id"] if gold is not None else ""
                candidate_id = stable_id(run_id, did, sid, phenomenon, p_id, g_id)
                candidates.append({"candidate_id": candidate_id, "run_id": run_id, "method_name": method_name,
                                   "split": evaluation["split"], "dialogue_id": did, "sentence_id": sid,
                                   "phenomenon": phenomenon, "fp_record_id": p_id, "fn_record_id": g_id,
                                   "association_status": association})
                if pre is not None:
                    records[("FP", pre)]["candidate_id"] = candidate_id
                if gold is not None:
                    records[("FN", gold)]["candidate_id"] = candidate_id
            if local_text and (fp or fn):
                text = row["sentence"]

                def private_entity(entity):
                    return {"type": entity.entity_type, "start": entity.start, "end": entity.end,
                            "text": text[entity.start:entity.end]}

                local_rows.append({"run_id": run_id, "dialogue_id": did, "sentence_id": sid,
                                   "sentence": text, "gold_entities": [private_entity(e) for e in sorted(g, key=entity_key)],
                                   "predicted_entities": [private_entity(e) for e in sorted(p, key=entity_key)]})
    actual = score_entity_sets(gold_all, pred_all)
    if actual != evaluation["entities"]:
        raise ValueError("Error extraction does not match evaluation entity counts/metrics")
    summary = {"schema_version": "1.0", "purpose": evaluation["purpose"], "run_id": run_id,
               "method_name": method_name, "status": "PASS", "inputs": evaluation["inputs"],
               "entities": actual, "rows": dict(Counter(r["category"] for r in errors + matches)),
               "candidate_counts": dict(Counter(r["phenomenon"] for r in candidates)),
               "ambiguous_unlinked_rows": sum(r["association_status"] == "ambiguous_unlinked" for r in candidates),
               "local_text_exported": local_text, "notice": "Candidate phenomena are not causal explanations and never change TP/FP/FN."}
    for kind, metric in actual["per_type"].items():
        for category, key in (("TP", "tp"), ("FP", "fp"), ("FN", "fn")):
            if sum(r["entity_type"] == kind and r["category"] == category for r in errors + matches) != metric[key]:
                raise ValueError("Per-type exported record count mismatch")
    return errors, matches, candidates, summary, local_rows


def export_errors(evaluation, data, predictions, directory, *, run_id, method_name, local_text=False, bom=False):
    details = build_details(evaluation, data, predictions, run_id=run_id, method_name=method_name, local_text=local_text)
    errors, matches, candidates, summary, local_rows = details
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_csv(directory / "entity_errors.csv", FIELDS, errors, bom=bom)
    write_csv(directory / "entity_matches.csv", FIELDS, matches, bom=bom)
    write_csv(directory / "error_candidates.csv", CANDIDATE_FIELDS, candidates, bom=bom)
    write_json(directory / "error_summary.json", summary)
    if local_text:
        private = directory / "local_analysis"
        private.mkdir()
        with (private / "sentences.jsonl").open("x", encoding="utf-8") as stream:
            for row in local_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    verify_error_files(evaluation, directory)
    return summary


def verify_error_files(evaluation, directory):
    """Reopen saved CSVs, verify count identities and one-to-one associations."""
    directory = Path(directory)
    rows = []
    for name in ("entity_errors.csv", "entity_matches.csv"):
        with (directory / name).open(encoding="utf-8-sig", newline="") as stream:
            saved = list(csv.DictReader(stream))
        allowed = ("TP",) if name == "entity_matches.csv" else ("FP", "FN")
        if any(r["category"] not in allowed or r["entity_type"] not in ENTITY_TYPES for r in saved):
            raise ValueError("invalid category/type in exported CSV")
        rows.extend(saved)
    ids = [r["record_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate exported entity record ID")
    for kind, metric in evaluation["entities"]["per_type"].items():
        for category, key in (("TP", "tp"), ("FP", "fp"), ("FN", "fn")):
            if sum(r["entity_type"] == kind and r["category"] == category for r in rows) != metric[key]:
                raise ValueError("saved CSV counts disagree with evaluation")
    with (directory / "error_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        candidates = list(csv.DictReader(stream))
    associated = [r[k] for r in candidates for k in ("fp_record_id", "fn_record_id") if r[k]]
    error_ids = {r["record_id"] for r in rows if r["category"] in ("FP", "FN")}
    if len(associated) != len(set(associated)) or set(associated) != error_ids:
        raise ValueError("candidate associations duplicate or omit error records")
    summary = read_json(directory / "error_summary.json")
    if summary["entities"] != evaluation["entities"] or summary["rows"] != dict(Counter(r["category"] for r in rows)):
        raise ValueError("saved error summary disagrees with evaluation")
    if summary["candidate_counts"] != dict(Counter(r["phenomenon"] for r in candidates)):
        raise ValueError("saved candidate counts disagree")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--local-text", action="store_true")
    parser.add_argument("--csv-bom", action="store_true")
    args = parser.parse_args()
    result = read_json(args.evaluation)
    verify_inputs(result)
    validation, data, pred = load_and_validate(result["inputs"]["data"]["path"], result["inputs"]["predictions"]["path"], split=result["split"])
    if validation["status"] != "PASS" or any(validation["inputs"][k]["sha256"] != result["inputs"][k]["sha256"] for k in ("data", "predictions")):
        raise ValueError("source changed or validation did not PASS")
    target = new_directory(args.output_dir, input_files=[args.evaluation, result["inputs"]["data"]["path"], result["inputs"]["predictions"]["path"]])
    export_errors(result, data, pred, target / "errors", run_id=args.run_id, method_name=args.method, local_text=args.local_text, bom=args.csv_bom)
    print("PASS", target / "errors")


if __name__ == "__main__":
    main()
