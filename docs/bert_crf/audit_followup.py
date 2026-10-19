"""Read-only follow-up checks. Writes only new evidence files, never sources."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import urllib.request
import urllib.error
import urllib.parse
import re
import zipfile

root = Path(__file__).resolve().parents[2]
evidence = Path(sys.argv[1]).resolve()
if not evidence.is_relative_to(root / "docs/bert_crf/evidence"):
    raise ValueError("Expected a stage-0 evidence directory")


def save(name, obj):
    with (evidence / name).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


if len(sys.argv) > 2 and sys.argv[2] == "--wheel-index":
    url = "https://download.pytorch.org/whl/cu124/torch/"
    try:
        with urllib.request.urlopen(url, timeout=25) as response:
            raw = response.read()
        names = [urllib.parse.unquote(x.split("/")[-1].split("#")[0])
                 for x in re.findall(r'href="([^"]+)', raw.decode())]
        wheels = [x for x in names if x.startswith("torch-2.6.0+") and "win_amd64" in x]
        obj = {"url": url, "index_sha256": hashlib.sha256(raw).hexdigest(),
               "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "torch_2_6_windows_wheels": wheels,
               "cp313_windows_available": any("cp313" in x for x in wheels),
               "scope": "Index metadata only; no wheel download or install"}
    except Exception as exc:
        obj = {"url": url, "error": str(exc), "status": "UNVERIFIED"}
    save("torch_wheel_index_check.json", obj)
    print(json.dumps(obj, indent=2))
    sys.exit(0)


stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
spec = "import sys,importlib.util,json; print(sys.executable); print(sys.version); print(json.dumps({m:bool(importlib.util.find_spec(m)) for m in ['torch','transformers','tokenizers','torchcrf','TorchCRF']},indent=2))"
args = ["C:/apps/anaconda/envs/mcmc_env/python.exe", "-X", "utf8", "-c", spec]
p = subprocess.run(args, capture_output=True, timeout=30)
save("secondary_environment.json", {"timestamp": stamp, "args": args, "returncode": p.returncode,
     "stdout": p.stdout.decode("utf-8"), "stderr": p.stderr.decode("utf-8")})

checks = []
req = evidence / "interface_snapshot/code/requirements.txt"
for line in req.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.startswith("#") or "==" not in line:
        continue
    name, version = line.strip().split("==")
    if name == "torch":
        # This local CUDA version is on the separate PyTorch index, not PyPI.
        continue
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    r = {"name": name, "version": version, "url": url, "checked_at": stamp}
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            obj = json.load(response)
        r.update(status="FOUND", requires_python=obj["info"].get("requires_python"),
                 requires_dist=obj["info"].get("requires_dist"),
                 files=[x["filename"] for x in obj["urls"]])
    except urllib.error.HTTPError as exc:
        r.update(status="HTTP_ERROR", http_status=exc.code)
    except Exception as exc:
        r.update(status="UNVERIFIED", error=str(exc))
    checks.append(r)
    print(name, version, r["status"], r.get("http_status", ""), flush=True)
save("dependency_index_checks.json", checks)

with zipfile.ZipFile(root / "IMCS-V2-NER.zip") as z:
    mappings = json.loads(z.read("IMCS-V2-NER/mappings.json"))
    data = json.loads(z.read("IMCS-V2-NER/IMCS-V2_train.json"))
    n_bad = 0
    examples = []
    dup = 0
    empty = 0
    whitespace = 0
    nonbmp = 0
    for did, d in data.items():
        ids = [s["sentence_id"] for s in d["dialogue"]]
        dup += len(ids) - len(set(ids))
        for s in d["dialogue"]:
            text = s["sentence"]
            empty += not text
            whitespace += any(c.isspace() for c in text)
            nonbmp += any(ord(c) > 0xffff for c in text)
            prev = "O"
            for i, tag in enumerate(s["BIO_label"].split(" ")):
                if tag.startswith("I-") and prev not in ("B-" + tag[2:], tag):
                    n_bad += 1
                    if len(examples) < 8:
                        examples.append({"dialogue_id": did, "sentence_id": s["sentence_id"],
                                         "index": i, "previous_tag": prev, "tag": tag})
                prev = tag
    first = next(iter(data.values()))
    save("train_contract_checks.json", {
        "timestamp": stamp, "scope": "Official train only. No dev/test label parsing.",
        "dialogue_fields": sorted(first), "sentence_fields": sorted(first["dialogue"][0]),
        "mapping_bio2id": mappings[2], "mapping_id2bio": mappings[3],
        "illegal_BIO_transitions": n_bad, "illegal_BIO_examples_no_text": examples,
        "duplicate_sentence_ids_within_dialogue": dup, "empty_sentences": empty,
        "sentences_with_whitespace": whitespace, "sentences_with_nonbmp": nonbmp,
    })

unchanged = []
for x in json.loads((evidence / "input_manifest.json").read_text(encoding="utf-8")):
    actual = hashlib.sha256((root / x["path"]).read_bytes()).hexdigest()
    unchanged.append({"path": x["path"], "unchanged": actual == x["sha256"]})
for x in json.loads((evidence / "interface_manifest.json").read_text(encoding="utf-8")):
    actual = hashlib.sha256((evidence / "interface_snapshot" / x["path"]).read_bytes()).hexdigest()
    unchanged.append({"path": "interface_snapshot/" + x["path"], "unchanged": actual == x["sha256"]})
save("source_integrity_check.json", {"timestamp": stamp, "files": unchanged})
print("Saved follow-up evidence:", evidence)
