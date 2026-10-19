"""Stage-0 evidence collector. Standard library only; no training or installs.

Existing evidence directories are never reused. Dataset JSON is read in memory;
dev/test labels are not parsed. Interface sources are extracted verbatim.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import urllib.request
import zipfile

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime.now(ZoneInfo("Asia/Shanghai"))
OUT = ROOT / "docs" / "bert_crf" / "evidence" / NOW.strftime("%Y%m%dT%H%M%S%z")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def command(args, cwd=ROOT, env=None):
    begin = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    t0 = time.perf_counter()
    try:
        p = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=60)
        result = dict(returncode=p.returncode,
                      stdout=p.stdout.decode("utf-8", errors="replace"),
                      stderr=p.stderr.decode("utf-8", errors="replace"))
    except (OSError, subprocess.TimeoutExpired) as exc:
        result = {"execution_error": str(exc)}
    result.update(args=args, cwd=str(cwd), started_at=begin,
                  elapsed_seconds=time.perf_counter()-t0)
    write_json(OUT / "commands" / f"{len(COMMANDS):02d}.json", result)
    COMMANDS.append(result)
    print(json.dumps({"command": args, "returncode": result.get("returncode"),
                      "error": result.get("execution_error")}, ensure_ascii=False))
    return result


COMMANDS = []


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = []
    for p in sorted(ROOT.iterdir()):
        if p.is_file():
            manifest.append({"path": p.name, "bytes": p.stat().st_size,
                             "sha256": sha(p.read_bytes())})
    write_json(OUT / "input_manifest.json", manifest)
    interface_zip = next(ROOT.glob("BERT*.zip"))
    snapshot = OUT / "interface_snapshot"
    with zipfile.ZipFile(interface_zip) as z:
        entries = []
        for item in z.infolist():
            dest = snapshot / item.filename
            if not dest.resolve().is_relative_to(snapshot.resolve()):
                raise ValueError("Archive path escapes snapshot")
            if item.is_dir():
                continue
            raw = z.read(item)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            entries.append({"path": item.filename, "bytes": len(raw), "sha256": sha(raw)})
    write_json(OUT / "interface_manifest.json", entries)
    split = json.loads((snapshot / "code/artifacts/split_v1.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(ROOT / "IMCS-V2-NER.zip") as z:
        archive_manifest = []
        for item in z.infolist():
            if not item.is_dir():
                raw = z.read(item)
                archive_manifest.append({"path": item.filename, "bytes": len(raw), "sha256": sha(raw)})
        raw = z.read("IMCS-V2-NER/IMCS-V2_train.json")
        data = json.loads(raw)
        tr, va = split["train"], split["val"]
        check = {
            "source_sha256_actual": sha(raw),
            "source_sha256_matches": sha(raw) == split["source_sha256"],
            "train_ids_unique": len(tr) == len(set(tr)),
            "val_ids_unique": len(va) == len(set(va)),
            "disjoint": set(tr).isdisjoint(va),
            "exact_official_train_coverage": set(tr) | set(va) == set(data),
            "version": split["version"], "seed": split["seed"],
            "recorded_stats": split["stats"],
            "actual_counts": {},
            "dev_test_policy": "Only archive bytes hashed; dev/test JSON labels not parsed.",
        }
        allowed = {"O"} | {f"{p}-{e}" for p in ("B", "I") for e in
                             ("Symptom", "Drug", "Drug_Category", "Medical_Examination", "Operation")}
        for name, ids in (("train", tr), ("val", va)):
            sentences = [s for did in ids for s in data[did]["dialogue"]]
            check["actual_counts"][name] = {
                "dialogues": len(ids), "sentences": len(sentences),
                "chars": sum(len(s["sentence"]) for s in sentences),
                "length_mismatches": sum(len(s["sentence"]) != len(s["BIO_label"].split(" ")) for s in sentences),
                "unknown_labels": sorted({t for s in sentences for t in s["BIO_label"].split(" ")} - allowed),
                "non_string_sentence_ids": sum(not isinstance(s["sentence_id"], str) for s in sentences),
                "max_sentence_chars": max(len(s["sentence"]) for s in sentences),
            }
        write_json(OUT / "dataset_archive_manifest.json", archive_manifest)
        write_json(OUT / "split_verification.json", check)
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    write_json(OUT / "environment.json", {
        "timestamp": NOW.isoformat(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(),
        "modules_available": {m: importlib.util.find_spec(m) is not None for m in
                              ("torch", "transformers", "tokenizers", "torchcrf", "TorchCRF", "numpy", "pytest", "seqeval")},
        "demo_environment_overrides": {k: env[k] for k in
            ("PYTHONUTF8", "PYTHONIOENCODING", "PYTHONDONTWRITEBYTECODE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")},
    })
    command(["git", "status", "--short"])
    command(["git", "log", "-5", "--oneline"])
    command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv"])
    command([sys.executable, "-m", "pip", "list", "--format=json"])
    command([sys.executable, "-X", "utf8", "-m", "scripts.demo_dataloader"], snapshot / "code", env)
    command([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-v"], snapshot / "code", env)

    # Direct probes of existing pure functions, without torch/tokenizer substitutes.
    sys.path.insert(0, str(snapshot / "code"))
    from src.align import gold_to_token_labels, tokens_to_char_labels
    from src.labels import LABEL2ID
    probes = {}
    probes["zero_width_ignored_continuation_not_ignored"] = gold_to_token_labels(
        ["B-Drug", "I-Drug"], [(0, 0), (0, 1), (1, 2), (0, 0)])
    probes["multichar_boundary_loss"] = {
        "gold": ["O", "B-Drug"],
        "restored": tokens_to_char_labels(gold_to_token_labels(["O", "B-Drug"], [(0, 2)]), [(0, 2)], 2)}
    probes["uncovered_filled_O"] = tokens_to_char_labels([LABEL2ID["O"]], [(0, 1)], 3)
    probes["illegal_leading_I_repaired"] = tokens_to_char_labels([LABEL2ID["I-Drug"]], [(0, 1)], 1)
    probes["zip_length_mismatch_not_rejected"] = tokens_to_char_labels([LABEL2ID["O"]], [(0, 1), (1, 2)], 2)
    try:
        tokens_to_char_labels([-100], [(0, 1)], 1)
    except Exception as exc:
        probes["ignore_index_at_nonzero_span"] = {"exception": type(exc).__name__, "message": str(exc)}
    write_json(OUT / "pure_alignment_probes.json", {
        "scope": "Synthetic inputs passed to unchanged functions. No real tokenizer or DataLoader test.",
        "observations": probes})

    # Pin upstream references to a real commit. Never fetch datasets/model weights.
    headers = {"User-Agent": "NLPexp-stage0-audit", "Accept": "application/vnd.github+json"}
    def fetch(url):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=25) as response:
            return response.read()
    try:
        repo = "https://api.github.com/repos/lemuria-wchen/imcs21-cblue"
        commit = json.loads(fetch(repo + "/commits/main"))
        revision = commit["sha"]
        tree = json.loads(fetch(repo + f"/git/trees/{revision}?recursive=1"))
        paths = [i["path"] for i in tree["tree"] if i["type"] == "blob" and
                 (i["path"].startswith("task1/BERT-NER/") or
                  i["path"] in ("task1/README.md", "README.md", "LICENSE"))]
        files = []
        for path in paths:
            if not path.endswith((".py", ".md", ".txt", ".json", ".sh")) and path != "LICENSE":
                continue
            raw = fetch(f"https://raw.githubusercontent.com/lemuria-wchen/imcs21-cblue/{revision}/{path}")
            dest = OUT / "upstream_snapshot" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            files.append({"path": path, "bytes": len(raw), "sha256": sha(raw)})
        write_json(OUT / "upstream_manifest.json", {"repository": repo, "commit": revision,
                   "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                   "files": files, "tree_paths": paths})
    except Exception as exc:
        write_json(OUT / "upstream_fetch_error.json", {"type": type(exc).__name__, "message": str(exc)})
    write_json(OUT / "commands_index.json", COMMANDS)
    print("EVIDENCE", OUT)


if __name__ == "__main__":
    main()
