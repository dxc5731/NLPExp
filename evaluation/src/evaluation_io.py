"""Small shared stdlib helpers for stage-three artifacts; never copy predictions."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

if __package__:
    from .validate_predictions import Report, parse_json
    from .ner_metrics import ENTITY_TYPES, LABEL_ORDER, prf_counts, score_confusion_matrix
else:
    from validate_predictions import Report, parse_json
    from ner_metrics import ENTITY_TYPES, LABEL_ORDER, prf_counts, score_confusion_matrix


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    raw = Path(path).read_bytes()
    report = Report("dev", False, 10)
    value = parse_json(raw, report, "data")
    if report.counts:
        raise ValueError("Invalid JSON/UTF-8: " + ", ".join(report.counts))
    return value


def resolve_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a nonempty string")
    path = Path(value)
    return (path if path.is_absolute() else Path(base) / path).resolve()


def new_directory(root, *, input_files=(), data_directories=()):
    root = Path(root).resolve()
    if any(root == Path(p).resolve() or root.is_relative_to(Path(p).resolve()) for p in input_files):
        raise ValueError("output directory cannot overwrite or descend from an input file")
    if any(root == Path(p).resolve() or root.is_relative_to(Path(p).resolve()) for p in data_directories):
        raise ValueError("output directory cannot be inside a supplied data directory")
    root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S%f") + "_"
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_cell(value):
    # Quoting alone does not stop Excel from interpreting formulas.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    if isinstance(value, str) and value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def write_csv(path, fields, rows, *, bom=False):
    with Path(path).open("x", newline="", encoding="utf-8-sig" if bom else "utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in fields})


def verify_inputs(evaluation):
    inputs = evaluation["inputs"]
    data = inputs["data"]
    path = Path(data["path"])
    if data.get("member"):
        if sha256(path) != data["archive_sha256"]:
            raise ValueError("source archive hash changed")
        with zipfile.ZipFile(path) as archive:
            infos = [i for i in archive.infolist() if i.filename == data["member"]]
            if len(infos) != 1:
                raise ValueError("ambiguous source archive member")
            digest = hashlib.sha256(archive.read(infos[0])).hexdigest()
    else:
        digest = sha256(path)
    if digest != data["sha256"]:
        raise ValueError("gold source hash changed")
    pred = inputs["predictions"]
    if sha256(pred["path"]) != pred["sha256"]:
        raise ValueError("prediction hash changed")


def verify_metrics(result):
    """Use existing formulas for arithmetic self-checks, not another scorer."""
    if result.get("status") != "SCORED" or result["validation"]["status"] != "PASS":
        raise ValueError("evaluation must be SCORED with validation PASS")
    entities = result["entities"]
    if list(entities["per_type"]) != list(ENTITY_TYPES):
        raise ValueError("entity types or order differ")
    for metric in entities["per_type"].values():
        numbers = [metric[k] for k in ("tp", "fp", "fn")]
        if any(type(n) is not int or n < 0 for n in numbers) or metric != prf_counts(*numbers):
            raise ValueError("invalid per-type entity metric counts/formula")
    pooled = prf_counts(*(sum(m[k] for m in entities["per_type"].values()) for k in ("tp", "fp", "fn")))
    if pooled != entities["micro"] or pooled["support"] != entities["gold_total"] or pooled["predicted"] != entities["predicted_total"]:
        raise ValueError("entity micro/count self-check failed")
    chars = result["characters"]
    cm = chars["confusion_matrix"]
    if cm["labels"] != list(LABEL_ORDER) or cm["rows"] != "gold" or cm["columns"] != "predicted":
        raise ValueError("character matrix policy differs")
    if chars != score_confusion_matrix(cm["counts"]) or chars["total"] != result["counts"]["characters"]:
        raise ValueError("character metric self-check failed")
