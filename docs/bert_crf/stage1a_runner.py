"""Isolated stage-1A setup and logged subprocess runner; never trains models.

All raw subprocess output is local-only because the supplied demo prints text.
Usage: python -X utf8 docs/bert_crf/stage1a_runner.py prepare
       python -X utf8 docs/bert_crf/stage1a_runner.py run [--cwd PATH] NAME -- ARGV...
       python -X utf8 docs/bert_crf/stage1a_runner.py tokenizer
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / ".local/bert_crf/stage1a_state.json"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "NLPexp-stage1a-audit"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def prepare():
    if STATE.exists():
        raise RuntimeError("Stage1A already prepared; reuse its state")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%z")
    evidence = ROOT / "docs/bert_crf/evidence" / (stamp + "_stage1a")
    local = ROOT / ".local/bert_crf" / (stamp + "_stage1a")
    evidence.mkdir(parents=True, exist_ok=False)
    local.mkdir(parents=True, exist_ok=False)
    state = {"started_at": now(), "run_id": stamp + "_stage1a", "evidence": str(evidence),
             "local": str(local), "interface": str(local / "interface"),
             "data": str(local / "data"), "tokenizer": str(local / "tokenizer"),
             "python": str(ROOT / ".venv-bert-crf/Scripts/python.exe")}
    dump(STATE, state)
    dump(evidence / "run_context.json", state)
    archive = next(ROOT.glob("BERT*.zip"))
    with zipfile.ZipFile(archive) as z:
        manifest = []
        for item in z.infolist():
            if item.is_dir():
                continue
            dest = Path(state["interface"]) / item.filename
            assert dest.resolve().is_relative_to(Path(state["interface"]).resolve())
            raw = z.read(item)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            manifest.append({"path": item.filename, "sha256": hashlib.sha256(raw).hexdigest()})
    dump(evidence / "interface_manifest.json", manifest)
    with zipfile.ZipFile(ROOT / "IMCS-V2-NER.zip") as z:
        raw = z.read("IMCS-V2-NER/IMCS-V2_train.json")
        target = Path(state["data"]) / "IMCS-V2_train.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(raw)
    dump(evidence / "input_identity.json", {
        "data_archive_sha256": hashlib.sha256((ROOT / "IMCS-V2-NER.zip").read_bytes()).hexdigest(),
        "interface_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "train_source_sha256": hashlib.sha256(raw).hexdigest(),
        "split_sha256": hashlib.sha256((Path(state["interface"]) / "code/artifacts/split_v1.json").read_bytes()).hexdigest(),
        "extracted_dataset_members": ["IMCS-V2-NER/IMCS-V2_train.json"],
        "dev_test_policy": "Not extracted or parsed in stage1A"})
    # Save actual official channel responses, not remembered install advice.
    page_url = "https://pytorch.org/get-started/locally/"
    html = get(page_url).decode()
    release = json.loads(get("https://pypi.org/pypi/torch/json"))["info"]["version"]
    index_url = "https://download.pytorch.org/whl/cu132/torch/"
    index = get(index_url)
    wheels = [urllib.parse.unquote(x.split("/")[-1].split("#")[0])
              for x in re.findall(r'href="([^"]+)', index.decode())]
    matches = [x for x in wheels if x.startswith(f"torch-{release}+cu132-") and "cp313-cp313-win_amd64" in x]
    if not matches:
        raise RuntimeError("No stable cp313 CUDA wheel found; inspect channel before choosing another")
    dump(evidence / "installation_plan.json", {
        "checked_at": now(), "python_strategy": "venv from existing CPython 3.13.5; no system site packages",
        "official_install_page": page_url, "page_sha256": hashlib.sha256(html.encode()).hexdigest(),
        "official_stable_metadata": "https://pypi.org/pypi/torch/json", "torch_stable_version": release,
        "wheel_index": index_url, "wheel_index_sha256": hashlib.sha256(index).hexdigest(),
        "selected_wheels": matches, "torch_requirement": f"torch=={release}+cu132",
        "selection_reason": "Official stable CUDA channel; verify sm_120 architecture AND real kernels after install, not inferred from driver alone",
        "other_dependencies": "Reuse v1 exact pins except torch; add pytest. Record installed transitive versions.",
        "crf_dependency": "Deferred: not needed for data audit or GPU smoke test"})
    req = (Path(state["interface"]) / "code/requirements.txt").read_text(encoding="utf-8")
    lines = [line for line in req.splitlines() if line.strip() and not line.startswith("#") and not line.startswith("torch==")]
    (evidence / "requirements-stage1a.in").write_text("\n".join(lines + ["pytest"]) + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2))
    print("SELECTED", matches)


def run(args):
    s = json.loads(STATE.read_text(encoding="utf-8"))
    evidence, local = Path(s["evidence"]), Path(s["local"])
    meta_path = evidence / "commands.jsonl"
    index = len(meta_path.read_text(encoding="utf-8").splitlines()) if meta_path.exists() else 0
    log = local / "logs" / f"{index:02d}_{args.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = args.argv[1:] if args.argv and args.argv[0] == "--" else args.argv
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               IMCS_DATA_DIR=s["data"], BERT_MODEL_DIR=s["tokenizer"],
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    begin = now()
    t0 = time.perf_counter()
    print("RUN", args.name, "LOG", log, flush=True)
    with log.open("xb") as output:
        output.write(("ARGV " + json.dumps(argv) + "\nCWD " + str(args.cwd or ROOT) + "\n").encode())
        output.flush()
        try:
            p = subprocess.run(argv, cwd=args.cwd or ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            code = p.returncode
            error = None
        except OSError as exc:
            code, error = 127, repr(exc)
            output.write(error.encode("utf-8"))
    entry = {"name": args.name, "argv": argv, "cwd": str(args.cwd or ROOT),
             "started_at": begin, "finished_at": now(), "elapsed_seconds": time.perf_counter()-t0,
             "exit_code": code, "error": error, "raw_log_local_only": str(log),
             "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
             "env_overrides": {k:env[k] for k in ("PYTHONUTF8","PYTHONIOENCODING","PYTHONDONTWRITEBYTECODE",
                                  "IMCS_DATA_DIR","BERT_MODEL_DIR","HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","TOKENIZERS_PARALLELISM")}}
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print("EXIT", code, "SECONDS", round(entry["elapsed_seconds"], 2), flush=True)
    return code


def tokenizer():
    s = json.loads(STATE.read_text(encoding="utf-8"))
    repo = "google-bert/bert-base-chinese"
    api = f"https://huggingface.co/api/models/{repo}"
    info = json.loads(get(api))
    revision = info["sha"]
    allowed = {"config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt", "special_tokens_map.json"}
    files = []
    out = Path(s["tokenizer"])
    out.mkdir(parents=True, exist_ok=False)
    for entry in info["siblings"]:
        name = entry["rfilename"]
        if name not in allowed:
            continue
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        raw = get(url)
        (out / name).write_bytes(raw)
        files.append({"name": name, "url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    dump(Path(s["evidence"]) / "tokenizer_manifest.json", {
        "downloaded_at": now(), "repo_id": repo, "revision": revision, "api": api,
        "files": files, "weights_downloaded": False})
    print("TOKENIZER", repo, revision, "FILES", len(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="mode", required=True)
    subs.add_parser("prepare")
    subs.add_parser("tokenizer")
    p = subs.add_parser("run")
    p.add_argument("name")
    p.add_argument("--cwd")
    p.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "tokenizer":
        tokenizer()
    else:
        sys.exit(run(args))
