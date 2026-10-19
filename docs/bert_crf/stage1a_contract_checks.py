"""Additional real-input checks prompted by the observed I->B repairs.

Checks projected token-gold BIO legality before a constrained CRF, and unlabeled
input support. Does not implement CRF or alter any public code/gold labels.
"""
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
state = json.loads((ROOT / ".local/bert_crf/stage1a_state.json").read_text(encoding="utf-8"))
e = Path(state["evidence"])
sys.path.insert(0, str(Path(state["interface"]) / "code"))
from src.align import encode_sentence, gold_to_token_labels
from src.data_io import load_split
from src.dataset import build_loader, collect_examples, load_tokenizer
from src.labels import LABELS


def illegal(tags):
    result = []
    prev = "O"
    for i,t in enumerate(tags):
        if t.startswith("I-") and prev not in ("B-"+t[2:], t):
            result.append({"position":i,"previous_tag":prev,"tag":t})
        prev = t
    return result


tok = load_tokenizer()
data = load_split("train")
split_path = Path(state["interface"]) / "code/artifacts/split_v1.json"
split = json.loads(split_path.read_text(encoding="utf-8"))
counts,issues = {},[]
for name in ("train","val"):
    c = Counter(sentences=0, original_gold_illegal_transitions=0,
                projected_token_gold_illegal_transitions=0, projected_token_gold_illegal_sentences=0,
                wordpiece_continuations=0, wordpiece_continuations_ignored=0)
    for ex in collect_examples(data,split[name]):
        c["sentences"] += 1
        _,off = encode_sentence(tok,ex["text"])
        labels = gold_to_token_labels(ex["tags"],off)
        token_tags = [LABELS[t] for t in labels if t != -100]
        bad = illegal(token_tags)
        c["original_gold_illegal_transitions"] += len(illegal(ex["tags"]))
        c["projected_token_gold_illegal_transitions"] += len(bad)
        c["projected_token_gold_illegal_sentences"] += bool(bad)
        ids,_ = encode_sentence(tok,ex["text"])
        for token,label in zip(tok.convert_ids_to_tokens(ids),labels):
            if token.startswith("##"):
                c["wordpiece_continuations"] += 1
                c["wordpiece_continuations_ignored"] += label == -100
        if bad:
            issues.append({"split":name,"dialogue_id":ex["dialogue_id"],"sentence_id":ex["sentence_id"],
                           "invalid_token_transitions":bad})
    counts[name] = dict(c)
    print(name,json.dumps(dict(c)),flush=True)

examples = collect_examples(data,split["train"][:1],require_labels=False)
batch = next(iter(build_loader(examples,tok,batch_size=4,with_labels=False)))
assert "labels" not in batch
valid = batch["attention_mask"].bool() & (batch["offsets"][:,:,1] > batch["offsets"][:,:,0])
assert valid.any(1).all()
checks = []
for item in json.loads((e / "interface_manifest.json").read_text(encoding="utf-8")):
    actual = hashlib.sha256((Path(state["interface"]) / item["path"]).read_bytes()).hexdigest()
    checks.append({"path":item["path"],"unchanged":actual == item["sha256"]})
assert all(x["unchanged"] for x in checks)
old = ROOT / "docs/bert_crf/evidence/20261019T160537+0800/input_manifest.json"
original = json.loads(old.read_text(encoding="utf-8"))
assert all(hashlib.sha256((ROOT / x["path"]).read_bytes()).hexdigest() == x["sha256"] for x in original)
cfg = (ROOT / ".venv-bert-crf/pyvenv.cfg").read_text(encoding="utf-8")
assert "include-system-site-packages = false" in cfg
report = {"timestamp":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
          "reason_for_additional_check":"2 real I->B repairs require checking projected gold before constrained CRF training",
          "scope":"Official train only; original gold unchanged",
          "projected_gold_checks":counts,"projected_gold_illegal_examples_no_text":issues,
          "unlabeled_batch":{"fields":list(batch),"has_labels":"labels" in batch,
                             "input_shape":list(batch["input_ids"].shape),"valid_lengths_from_offsets":valid.sum(1).tolist()},
          "interface_integrity":checks,"original_six_inputs_unchanged":True,
          "venv_system_site_packages":False}
with (e / "contract_checks.json").open("x",encoding="utf-8") as f:
    json.dump(report,f,ensure_ascii=False,indent=2)
    f.write("\n")
print("CONTRACT CHECKS SAVED",e / "contract_checks.json")
