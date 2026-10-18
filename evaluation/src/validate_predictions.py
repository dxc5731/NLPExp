"""Read-only IMCS-V2-NER input validation. No inference, repair, or scoring.

Python 3.10+, standard library only. Run --help for CLI usage.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import zipfile


ENTITY_TYPES = (
    "Symptom", "Drug", "Drug_Category", "Medical_Examination", "Operation"
)
LABELS = frozenset(["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")])
LOAD_FAILED = object()
CHECK_NAMES = (
    "data_json", "data_structure", "gold_presence", "gold_format",
    "gold_labels", "gold_bio", "gold_alignment", "prediction_json",
    "prediction_structure", "id_coverage", "record_counts",
    "prediction_format", "prediction_labels", "prediction_bio",
    "prediction_alignment",
)


def safe(value, limit=200):
    """Keep reports small and encodable, including on malformed surrogate input."""
    if isinstance(value, str):
        value = value.encode("utf-8", errors="backslashreplace").decode("utf-8")
        return value if len(value) <= limit else value[:limit] + "..."
    return value


class Report:
    def __init__(self, split, audit_data, max_examples):
        self.split = split
        self.audit_data = audit_data
        self.max_examples = max_examples
        self.checks = {n: {"checked": 0, "errors": 0, "skipped": 0, "reasons": []}
                       for n in CHECK_NAMES}
        self.counts = Counter()
        self.examples = []
        self.example_counts = Counter()
        self.expected = set()
        self.failed = set()
        self.checked_sentences = set()
        self.inputs = {}
        self.data = {}
        self.predictions = None

    def checked(self, name, count=1):
        self.checks[name]["checked"] += count

    def skip(self, name, reason, count=1):
        check = self.checks[name]
        check["skipped"] += count
        if reason not in check["reasons"]:
            check["reasons"].append(reason)

    def error(self, check, code, message, *, source="data", dialogue_id=None,
              sentence_id=None, **details):
        self.checks[check]["errors"] += 1
        self.counts[code] += 1
        if source == "predictions" and dialogue_id is not None:
            if sentence_id is not None:
                self.failed.add((dialogue_id, sentence_id))
            else:
                self.failed.update(k for k in self.expected if k[0] == dialogue_id)
        if len(self.examples) >= self.max_examples or self.example_counts[code] >= 3:
            return
        location = {"source": source}
        if dialogue_id is not None:
            location["dialogue_id"] = safe(dialogue_id)
        if sentence_id is not None:
            location["sentence_id"] = safe(sentence_id)
        self.examples.append({"check": check, "code": code, "message": message,
                              "location": location,
                              **{k: safe(v) for k, v in details.items()}})
        self.example_counts[code] += 1

    def finish(self):
        required = ["data_json", "data_structure", "gold_presence"]
        if self.data.get("format") == "raw" and self.split != "test":
            required += ["gold_format", "gold_labels", "gold_bio", "gold_alignment"]
        if not self.audit_data:
            required += [n for n in CHECK_NAMES if n.startswith("prediction_")]
            required += ["id_coverage", "record_counts"]
        for name, check in self.checks.items():
            if check["errors"]:
                check["status"] = "FAIL"
            elif check["skipped"] or not check["checked"]:
                check["status"] = "NOT_RUN"
                if not check["reasons"]:
                    check["reasons"].append("not_applicable_or_dependency_unavailable")
            else:
                check["status"] = "PASS"
        if self.counts:
            status = "FAIL"
        elif any(self.checks[n]["status"] != "PASS" for n in required):
            status = "NOT_RUN"
        else:
            status = "PASS"
        if self.predictions is not None:
            failed = self.expected & self.failed
            passed = (self.expected & self.checked_sentences) - failed
            self.predictions["expected_sentences"] = {
                "passed": len(passed), "failed": len(failed),
                "not_fully_checked": len(self.expected - passed - failed),
            }
        return {
            "schema_version": "1.0", "purpose": "input_validation_only",
            "mode": "data_audit" if self.audit_data else "prediction_validation",
            "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
            "python_version": platform.python_version(), "split": self.split,
            "status": status,
            "character_unit": "Python len(str): Unicode code points, no normalization",
            "inputs": self.inputs, "data": self.data, "predictions": self.predictions,
            "required_checks": required, "checks": self.checks,
            "errors": {"total": sum(self.counts.values()),
                       "by_code": dict(sorted(self.counts.items())),
                       "examples": self.examples,
                       "examples_omitted": sum(self.counts.values()) - len(self.examples)},
        }


class ObjectPairs(list):
    """Distinguish JSON objects from arrays before duplicate keys are lost."""


def reject_constant(value):
    raise ValueError(f"Non-JSON constant: {value}")


def parse_json(raw, report, source):
    check = "data_json" if source == "data" else "prediction_json"
    report.checked(check)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        report.error(check, "INVALID_UTF8", "Input must be UTF-8.", source=source,
                     byte_offset=exc.start)
        return LOAD_FAILED
    if text.startswith("\ufeff"):
        report.error(check, "UTF8_BOM", "Use UTF-8 without BOM.", source=source)
        return LOAD_FAILED

    def convert(node, path=()):
        if isinstance(node, ObjectPairs):
            result = {}
            for key, value in node:
                child_path = path + (key,)
                converted = convert(value, child_path)
                if key in result:
                    location = {}
                    if source == "predictions":
                        location["dialogue_id"] = child_path[0]
                        if len(child_path) > 1:
                            location["sentence_id"] = child_path[1]
                    report.error(check, "DUPLICATE_JSON_KEY",
                                 "Duplicate key; file rejected. First value retained only for diagnostics.",
                                 source=source, json_pointer=pointer(child_path), **location)
                else:
                    result[key] = converted
                check_scalar(key, child_path)
            return result
        if isinstance(node, list):
            return [convert(value, path + (str(i),)) for i, value in enumerate(node)]
        if isinstance(node, str):
            check_scalar(node, path)
        return node

    def check_scalar(value, path):
        if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            report.error(check, "INVALID_UNICODE_SCALAR", "Unpaired surrogate is not a Unicode scalar.",
                         source=source, json_pointer=pointer(path))

    def pointer(path):
        return "/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in path)

    try:
        return convert(json.loads(text, object_pairs_hook=ObjectPairs,
                                  parse_constant=reject_constant))
    except (ValueError, RecursionError) as exc:
        details = {}
        if isinstance(exc, json.JSONDecodeError):
            details = {"line": exc.lineno, "column": exc.colno}
        report.error(check, "INVALID_JSON", "JSON parsing failed.", source=source, **details)
        return LOAD_FAILED


def data_candidates(split):
    names = [f"IMCS-V2_{split}.json"]
    if split == "test":
        names.append("IMCS-V2-NER_test.json")
    return names


def read_data(path, split, report):
    path = Path(path).resolve()
    report.inputs["data"] = {"requested_path": str(path)}
    try:
        if path.is_dir():
            chosen = None
            for name in data_candidates(split):
                candidates = [p for p in (path / name, path / "IMCS-V2-NER" / name) if p.is_file()]
                if len(candidates) > 1:
                    raise ValueError("AMBIGUOUS_DATA")
                if candidates:
                    chosen = candidates[0]
                    break
            if chosen is None:
                raise ValueError("DATA_NOT_FOUND")
            path = chosen.resolve()
        if path.suffix.lower() == ".zip":
            archive_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with zipfile.ZipFile(path) as archive:
                chosen = None
                for name in data_candidates(split):
                    candidates = [i for i in archive.infolist() if not i.is_dir()
                                  and PurePosixPath(i.filename).name == name]
                    if len(candidates) > 1:
                        raise ValueError("AMBIGUOUS_DATA")
                    if candidates:
                        chosen = candidates[0]
                        break
                if chosen is None:
                    raise ValueError("DATA_NOT_FOUND")
                raw = archive.read(chosen)
                report.inputs["data"].update(path=str(path), member=chosen.filename,
                                               archive_sha256=archive_hash)
        else:
            for other_split in ("train", "dev", "test"):
                if path.name in data_candidates(other_split) and other_split != split:
                    raise ValueError("SPLIT_MISMATCH")
            raw = path.read_bytes()
            report.inputs["data"]["path"] = str(path)
        report.inputs["data"]["sha256"] = hashlib.sha256(raw).hexdigest()
        return parse_json(raw, report, "data")
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        code = str(exc) if str(exc) in {"DATA_NOT_FOUND", "AMBIGUOUS_DATA", "SPLIT_MISMATCH"} else "DATA_READ_ERROR"
        report.error("data_json", code, "Cannot resolve/read a unique data source for this split.",
                     exception_type=type(exc).__name__)
        return LOAD_FAILED


def check_sequence(value, text, report, source, dialogue_id, sentence_id):
    prefix = "gold" if source == "data" else "prediction"
    location = dict(source=source, dialogue_id=dialogue_id, sentence_id=sentence_id)
    format_check = prefix + "_format"
    report.checked(format_check)
    if value and (value.startswith(" ") or value.endswith(" ") or "  " in value
                  or any(c.isspace() and c != " " for c in value)):
        report.error(format_check, "INVALID_SEPARATOR",
                     "Separate tags by exactly one ASCII space; no leading/trailing whitespace.", **location)
        for name in ("labels", "bio", "alignment"):
            report.skip(prefix + "_" + name, "invalid_separator")
        return False
    tags = value.split(" ") if value else []
    report.checked(prefix + "_labels")
    report.checked(prefix + "_bio")
    previous = "O"
    for index, tag in enumerate(tags):
        if tag not in LABELS:
            report.error(prefix + "_labels", "INVALID_LABEL", "Label is not in the 11-label set.",
                         tag_index=index, actual=tag, **location)
            report.skip(prefix + "_bio", "unknown_label_prevents_full_bio_check")
            # Unknown tags cannot authorize continuation of an entity.
            previous = None
            continue
        if tag.startswith("I-") and previous not in ("B-" + tag[2:], tag):
            report.error(prefix + "_bio", "ILLEGAL_I_TRANSITION",
                         "I-X must immediately follow B-X or I-X in the same sentence.",
                         tag_index=index, actual=tag, previous=previous, **location)
        previous = tag
    if text is None:
        report.skip(prefix + "_alignment", "original_sentence_text_unavailable")
        return False
    report.checked(prefix + "_alignment")
    if len(tags) != len(text):
        report.error(prefix + "_alignment", "LABEL_LENGTH_MISMATCH",
                     "Tag count must equal original Unicode code-point count.",
                     expected=len(text), actual=len(tags), **location)
    return True


def build_data(data, report):
    """Return original sentence coordinate map, never normalize IDs or text."""
    if data is LOAD_FAILED:
        report.skip("data_structure", "data_json_unavailable")
        return {}
    report.checked("data_structure")
    if not isinstance(data, dict) or not data:
        report.error("data_structure", "DATA_ROOT_TYPE", "Data must be a nonempty JSON object.")
        return {}
    is_raw = any(isinstance(v, dict) and "dialogue" in v for v in data.values())
    is_template = (report.split == "test" and not is_raw
                   and all(isinstance(v, dict) and v and all(x == "" for x in v.values())
                           for v in data.values()))
    if not is_raw and not is_template:
        report.error("data_structure", "DATA_FORMAT",
                     "Expected raw dialogue records; only test may use an empty ID template.")
        return {}
    texts = {}
    fields = set()
    sentence_fields = set()
    sentence_rows = 0
    gold_sentences = 0
    characters = 0
    for did, record in data.items():
        if not did or not isinstance(record, dict):
            report.error("data_structure", "DATA_DIALOGUE_TYPE", "Dialogue ID must be nonempty and value an object.",
                         dialogue_id=did)
            continue
        fields.update(record.keys())
        if is_template:
            for sid in record:
                sentence_rows += 1
                if not sid:
                    report.error("data_structure", "DATA_SENTENCE_ID_TYPE", "Sentence ID must be a nonempty string.",
                                 dialogue_id=did, sentence_id=sid)
                texts[(did, sid)] = None
            continue
        rows = record.get("dialogue")
        if not isinstance(rows, list) or not rows:
            report.error("data_structure", "DATA_DIALOGUE_ROWS", "dialogue must be a nonempty array.", dialogue_id=did)
            continue
        seen = set()
        for row_index, row in enumerate(rows):
            sentence_rows += 1
            if not isinstance(row, dict):
                report.error("data_structure", "DATA_SENTENCE_TYPE", "Sentence record must be an object.",
                             dialogue_id=did, row_index=row_index)
                continue
            sentence_fields.update(row.keys())
            sid = row.get("sentence_id")
            if not isinstance(sid, str) or not sid:
                report.error("data_structure", "DATA_SENTENCE_ID_TYPE", "sentence_id must be a nonempty string.",
                             dialogue_id=did, row_index=row_index)
                continue
            if sid in seen:
                report.error("data_structure", "DUPLICATE_SENTENCE_ID", "Repeated sentence_id inside one dialogue.",
                             dialogue_id=did, sentence_id=sid, row_index=row_index)
            seen.add(sid)
            text = row.get("sentence")
            if not isinstance(text, str):
                report.error("data_structure", "DATA_TEXT_TYPE", "sentence must be a string.",
                             dialogue_id=did, sentence_id=sid)
                text = None
            else:
                characters += len(text)
            # A duplicate source ID is a failure; use its first occurrence only for diagnostics.
            texts.setdefault((did, sid), text)
            report.checked("gold_presence")
            if "BIO_label" in row:
                gold_sentences += 1
            if report.split == "test":
                if "BIO_label" in row:
                    report.error("gold_presence", "UNEXPECTED_TEST_GOLD", "Official test should have no BIO gold.",
                                 dialogue_id=did, sentence_id=sid)
            elif not isinstance(row.get("BIO_label"), str):
                report.error("gold_presence", "MISSING_OR_INVALID_GOLD", "train/dev must contain a BIO_label string.",
                             dialogue_id=did, sentence_id=sid)
            else:
                check_sequence(row["BIO_label"], text, report, "data", did, sid)
    if is_template:
        report.skip("gold_presence", "empty_test_template_has_no_gold")
    report.data = {"format": "raw" if is_raw else "empty_test_template",
                   "dialogues": len(data), "sentences": sentence_rows,
                   "unique_sentence_ids": len(texts),
                   "sentences_with_text": sum(t is not None for t in texts.values()),
                   "characters": characters if is_raw else None,
                   "sentences_with_gold": gold_sentences,
                   "dialogue_fields": [safe(f) for f in sorted(fields)] if is_raw else [],
                   "sentence_fields": [safe(f) for f in sorted(sentence_fields)]}
    report.expected = set(texts)
    return texts


def validate_prediction_data(predictions, texts, report):
    if predictions is LOAD_FAILED:
        report.skip("prediction_structure", "prediction_json_unavailable")
        return
    report.checked("prediction_structure")
    if not isinstance(predictions, dict):
        report.error("prediction_structure", "PREDICTION_ROOT_TYPE", "Predictions must be a JSON object.", source="predictions")
        return
    actual = set()
    rows_count = 0
    reliable_source = bool(report.data) and not any(
        report.checks[n]["errors"] for n in ("data_json", "data_structure"))
    for did, rows in predictions.items():
        if not did or not isinstance(rows, dict):
            report.error("prediction_structure", "PREDICTION_DIALOGUE_TYPE", "Dialogue must map a nonempty ID to an object.",
                         source="predictions", dialogue_id=did)
            continue
        rows_count += len(rows)
        for sid, value in rows.items():
            key = (did, sid)
            actual.add(key)
            if not sid:
                report.error("prediction_structure", "PREDICTION_SENTENCE_ID_TYPE", "Sentence ID cannot be empty.",
                             source="predictions", dialogue_id=did, sentence_id=sid)
            if not isinstance(value, str):
                report.error("prediction_structure", "PREDICTION_VALUE_TYPE", "Sentence prediction must be a string.",
                             source="predictions", dialogue_id=did, sentence_id=sid,
                             actual_type=type(value).__name__)
                for name in ("format", "labels", "bio", "alignment"):
                    report.skip("prediction_" + name, "prediction_value_not_string")
                continue
            text = texts.get(key) if reliable_source else None
            full_check = check_sequence(value, text, report, "predictions", did, sid)
            if full_check and key in report.expected:
                report.checked_sentences.add(key)
    report.predictions = {"dialogues": len(predictions), "sentences": rows_count,
                          "unique_sentence_ids": len(actual),
                          "matched_sentence_ids": len(actual & report.expected)}
    if not reliable_source:
        report.skip("id_coverage", "original_data_structure_unavailable_or_ambiguous")
        report.skip("record_counts", "original_data_structure_unavailable_or_ambiguous")
        return
    report.checked("id_coverage")
    expected_dialogues = {k[0] for k in report.expected}
    for did in sorted(expected_dialogues - predictions.keys()):
        report.error("id_coverage", "MISSING_DIALOGUE", "Original dialogue is missing.", source="predictions", dialogue_id=did)
    for did in sorted(predictions.keys() - expected_dialogues):
        report.error("id_coverage", "EXTRA_DIALOGUE", "Dialogue ID is not in original data.", source="predictions", dialogue_id=did)
    for did, sid in sorted(report.expected - actual):
        report.error("id_coverage", "MISSING_SENTENCE", "Original sentence is missing.", source="predictions", dialogue_id=did, sentence_id=sid)
    for did, sid in sorted(actual - report.expected):
        report.error("id_coverage", "EXTRA_SENTENCE", "Sentence ID is not in original data.", source="predictions", dialogue_id=did, sentence_id=sid)
    report.checked("record_counts")
    if len(predictions) != report.data["dialogues"] or rows_count != report.data["sentences"]:
        report.error("record_counts", "RECORD_COUNT_MISMATCH", "Dialogue/sentence counts differ from original data.",
                     source="predictions", expected_dialogues=report.data["dialogues"], actual_dialogues=len(predictions),
                     expected_sentences=report.data["sentences"], actual_sentences=rows_count)


def load_and_validate(data_path, prediction_path=None, *, split, audit_data=False, max_examples=30):
    """Return (report, data, predictions) from the same read-only input snapshot.

    Consumers must require report['status'] == 'PASS' before using the objects.
    Reusing these parsed objects avoids rereading changed files after validation.
    """
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    if audit_data == (prediction_path is not None):
        raise ValueError("Use either audit_data=True or a prediction_path, exclusively")
    if max_examples < 0:
        raise ValueError("max_examples must be nonnegative")
    report = Report(split, audit_data, max_examples)
    data = read_data(data_path, split, report)
    texts = build_data(data, report)
    predictions = LOAD_FAILED
    if not audit_data:
        path = Path(prediction_path).resolve()
        report.inputs["predictions"] = {"path": str(path)}
        try:
            raw = path.read_bytes()
            report.inputs["predictions"]["sha256"] = hashlib.sha256(raw).hexdigest()
            predictions = parse_json(raw, report, "predictions")
        except OSError as exc:
            report.error("prediction_json", "PREDICTION_READ_ERROR", "Cannot read prediction file.",
                         source="predictions", exception_type=type(exc).__name__)
            predictions = LOAD_FAILED
        validate_prediction_data(predictions, texts, report)
    result = report.finish()
    # Template validation can establish IDs and BIO, but never character alignment.
    if result["data"].get("format") == "empty_test_template" and result["status"] == "PASS":
        result["status"] = "NOT_RUN"
    return result, data, predictions


def run_validation(data_path, prediction_path=None, *, split, audit_data=False, max_examples=30):
    """Backward-compatible report-only interface for the stage-one validator."""
    return load_and_validate(data_path, prediction_path, split=split,
                             audit_data=audit_data, max_examples=max_examples)[0]


def protected_report_path(report_path, data_path, prediction_path):
    """Never let a report overwrite inputs or a file inside the supplied data dir."""
    target = Path(report_path).resolve()
    data = Path(data_path).resolve()
    if target == data or (data.is_dir() and target.is_relative_to(data)):
        return True
    return prediction_path is not None and target == Path(prediction_path).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path,
                        help="Raw JSON, directory (direct or IMCS-V2-NER/), or original ZIP (read-only)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path, help="Internal predictions_dev.json / predictions_test.json")
    mode.add_argument("--audit-data", action="store_true", help="Audit source structure/gold only; no predictions or scores")
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--report", required=True, type=Path, help="Output JSON report, outside inputs")
    parser.add_argument("--max-examples", type=int, default=30, help="Maximum stored errors (up to 3 per code), default 30")
    args = parser.parse_args(argv)
    if args.max_examples < 0:
        parser.error("--max-examples must be nonnegative")
    if protected_report_path(args.report, args.data, args.predictions):
        parser.error("--report cannot overwrite inputs or be inside the supplied data directory")
    result = run_validation(args.data, args.predictions, split=args.split,
                            audit_data=args.audit_data, max_examples=args.max_examples)
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"FAIL: cannot write report ({type(exc).__name__}).", file=sys.stderr)
        return 2
    print(f"{result['status']} mode={result['mode']} split={args.split} "
          f"dialogues={result['data'].get('dialogues', '?')} "
          f"sentences={result['data'].get('sentences', '?')} errors={result['errors']['total']}")
    if result["errors"]["by_code"]:
        print("Errors: " + ", ".join(f"{k}={v}" for k, v in result["errors"]["by_code"].items()))
    unavailable = [n for n in result["required_checks"] if result["checks"][n]["status"] == "NOT_RUN"]
    if unavailable:
        print("NOT_RUN: " + ", ".join(unavailable))
    print(f"Report: {args.report.resolve()}")
    return {"PASS": 0, "FAIL": 1, "NOT_RUN": 2}[result["status"]]


if __name__ == "__main__":
    sys.exit(main())
