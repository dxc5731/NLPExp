"""Audit the unchanged public interface with real data and a real tokenizer.

No model, CRF implementation, predictions or dev/test labels are used.
JSON summaries/IDs contain no medical text. Detailed examples stay in .local/.
"""
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
STATE = json.loads((ROOT / ".local/bert_crf/stage1a_state.json").read_text(encoding="utf-8"))
EVIDENCE = Path(STATE["evidence"])
LOCAL = Path(STATE["local"])
sys.path.insert(0, str(Path(STATE["interface"]) / "code"))
from src.align import encode_sentence, gold_to_token_labels, tokens_to_char_labels
from src.config import MAX_LEN
from src.data_io import load_split
from src.dataset import build_loader, collect_examples, load_tokenizer
from src.labels import LABELS, LABEL2ID


def dump(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def lengths(values):
    return {"min": min(values), "max": max(values), "mean": float(np.mean(values)),
            "quantile_method": "numpy.linear",
            "percentiles": {str(p): float(np.percentile(values, p)) for p in (50,90,95,99,100)},
            "histogram_exact": dict(sorted(Counter(values).items()))}


def raw_expand(token_labels, offsets, n):
    """Instrument pre-fill/pre-repair states; compare final output to public helper."""
    out = [None] * n
    covered = [0] * n
    for label, (start, end) in zip(token_labels, offsets):
        if start == end:
            continue
        if not (0 <= start < end <= n):
            raise ValueError(f"Invalid offset {(start,end)} for length {n}")
        tag = LABELS[label]
        for k in range(start, end):
            covered[k] += 1
            out[k] = "I-" + tag[2:] if k > start and tag.startswith("B-") else tag
    return out, covered


def inspect(text, tags, tok, enc=None):
    ids, offsets = encode_sentence(tok, text, MAX_LEN) if enc is None else (enc["input_ids"], enc["offsets"])
    full = tok(text, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
    labels = gold_to_token_labels(tags, offsets)
    raw, coverage = raw_expand(labels, offsets, len(text))
    full_labels = gold_to_token_labels(tags, full["offset_mapping"])
    _, full_coverage = raw_expand(full_labels, full["offset_mapping"], len(text))
    restored, uncovered = tokens_to_char_labels(labels, offsets, len(text))
    filled = ["O" if t is None else t for t in raw]
    repairs = [k for k, (before, after) in enumerate(zip(filled, restored)) if before != after]
    expected = []
    for tag in filled:
        if tag.startswith("I-") and (not expected or expected[-1] not in ("B-"+tag[2:], tag)):
            tag = "B-"+tag[2:]
        expected.append(tag)
    assert expected == restored
    assert uncovered == [k for k,c in enumerate(coverage) if c == 0]
    mismatch = [k for k,(g,p) in enumerate(zip(tags,restored)) if g != p]
    lost = [k for k,(before,after) in enumerate(zip(full_coverage,coverage)) if before > 0 and after == 0]
    spans = []
    start = None
    for k,tag in enumerate(tags + ["O"]):
        if start is not None and not (tag.startswith("I-") and tag[2:] == tags[start][2:]):
            spans.append((start,k,tags[start][2:]))
            start = None
        if tag.startswith("B-"):
            start = k
    affected_entities = sum(any(c == 0 for c in coverage[s:e]) for s,e,_ in spans)
    cross_trunc = sum(any(k in lost for k in range(s,e)) and any(coverage[k] for k in range(s,e)) for s,e,_ in spans)
    multi = [(i,s,e) for i,(s,e) in enumerate(offsets) if e-s > 1]
    lossy = []
    reason_counts = Counter()
    for i,s,e in multi:
        tag = LABELS[labels[i]]
        projected = [tag] + [("I-"+tag[2:]) if tag.startswith("B-") else tag] * (e-s-1)
        if projected != tags[s:e]:
            reasons = []
            if any(t.startswith("B-") for t in tags[s+1:e]):
                reasons.append("entity_start_inside_token")
            if any(tags[k] == "O" and tags[k-1] != "O" for k in range(s+1,e)):
                reasons.append("entity_end_inside_token")
            if any(tags[k] != "O" and tags[k-1] != "O" and tags[k][2:] != tags[k-1][2:] for k in range(s+1,e)):
                reasons.append("entity_type_change_inside_token")
            if not reasons:
                reasons.append("other_unrepresentable_span")
            reason_counts.update(reasons)
            lossy.append({"token_index": i, "span": [s,e], "gold": tags[s:e],
                          "projected": projected, "reasons": reasons})
    valid = [t != -100 for t in labels]
    positions = [i for i,v in enumerate(valid) if v]
    internal_ignored = sum(not valid[i] for i in range(positions[0],positions[-1]+1)) if positions else 0
    real_special = [i for i,(s,e) in enumerate(offsets) if s == e]
    return {
        "n_chars":len(text), "n_tokens_with_special":len(ids), "n_full_tokens_with_special":len(full["input_ids"]),
        "uncovered":uncovered, "overcovered":[i for i,c in enumerate(coverage) if c > 1],
        "full_uncovered":[i for i,c in enumerate(full_coverage) if c == 0], "truncated_chars":lost,
        "needs_truncation":len(full["input_ids"]) > MAX_LEN, "affected_entities":affected_entities,
        "entities_crossing_truncation":cross_trunc, "mismatch":mismatch,
        "repairs":repairs, "repair_pairs":[[filled[k],restored[k]] for k in repairs],
        "multichar_tokens":len(multi), "lossy_tokens":lossy, "loss_reasons":dict(reason_counts),
        "unk_tokens":ids.count(tok.unk_token_id),
        "wordpiece_continuation_tokens":sum(t.startswith("##") for t in tok.convert_ids_to_tokens(ids)),
        "internal_ignored_positions":internal_ignored,
        "special_positions":len(real_special),
        "special_labels_wrong":sum(labels[i] != -100 for i in real_special),
        "labels":labels, "offsets":offsets, "restored":restored,
    }


def batch_check(loader, split):
    b = next(iter(loader))
    tensors = {k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in b.items() if torch.is_tensor(v)}
    by_offsets = b["attention_mask"].bool() & (b["offsets"][:,:,1] > b["offsets"][:,:,0])
    assert torch.equal(by_offsets, b["labels"] != -100)
    selected = [torch.where(row)[0] for row in by_offsets]
    n = [len(x) for x in selected]
    packed = torch.zeros((len(n), max(n)), dtype=torch.long)
    mask = torch.arange(max(n))[None,:] < torch.tensor(n)[:,None]
    roundtrip = []
    for i,idx in enumerate(selected):
        packed[i,:len(idx)] = b["labels"][i,idx]
        roundtrip.append(torch.equal(packed[i,:len(idx)],b["labels"][i,idx]))
    assert (packed >= 0).all() and (packed < len(LABELS)).all()
    assert mask[:,0].all()
    assert not ((~mask[:,:-1]) & mask[:,1:]).any()
    assert all(roundtrip)
    for row, ex in enumerate(loader.dataset.examples[:len(n)]):
        assert b["dialogue_id"][row] == ex["dialogue_id"] and b["sentence_id"][row] == ex["sentence_id"]
        assert b["text"][row] == ex["text"]
    result = {"split":split, "dataset_sentences":len(loader.dataset), "fields":list(b),
              "tensors":tensors, "metadata_list_lengths":{k:len(v) for k,v in b.items() if isinstance(v,list)},
              "sample_ids":[{"dialogue_id":d,"sentence_id":s} for d,s in zip(b["dialogue_id"],b["sentence_id"])],
              "attention_lengths":b["attention_mask"].sum(1).tolist(),
              "valid_label_lengths":by_offsets.sum(1).tolist(),
              "ignored_labels":int((b["labels"] == -100).sum()),
              "padding_positions":int((b["attention_mask"] == 0).sum()),
              "ignored_special_positions":int(((b["attention_mask"] == 1) & (~by_offsets)).sum()),
              "attention_equals_label_valid":torch.equal(b["attention_mask"].bool(),by_offsets),
              "raw_crf_mask_first_column":by_offsets[:,0].tolist(),
              "packing_probe":{"shape":list(packed.shape),"lengths":n,"bool_mask_prefix_contiguous":True,
                               "all_label_values_legal_including_padding":True,"selected_label_roundtrip":all(roundtrip)},
              "note":"Packing probe only; no CRF model or likelihood implemented"}
    print("BATCH",json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    t0 = time.perf_counter()
    tok = load_tokenizer()
    assert tok.is_fast
    split_path = Path(STATE["interface"]) / "code/artifacts/split_v1.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    source = Path(STATE["data"]) / "IMCS-V2_train.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == split["source_sha256"]
    data = load_split("train")
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]) | set(split["val"]) == set(data)
    summaries, batches = {}, []
    all_issues = []
    private_path = LOCAL / "alignment_issues_private.jsonl"
    with private_path.open("x",encoding="utf-8") as private:
        for name in ("train","val"):
            examples = collect_examples(data,split[name])
            loader = build_loader(examples,tok,batch_size=4,shuffle=False,num_workers=0)
            batches.append(batch_check(loader,name))
            counts = Counter()
            reasons = Counter()
            char_lens,token_lens = [],[]
            affected_ids = []
            for index,(ex,enc) in enumerate(zip(examples,loader.dataset.encoded)):
                item = inspect(ex["text"],ex["tags"],tok,enc)
                counts["sentences"] += 1
                counts["chars"] += item["n_chars"]
                counts["tokens_with_special"] += item["n_tokens_with_special"]
                char_lens.append(item["n_chars"])
                token_lens.append(item["n_full_tokens_with_special"])
                for key in ("multichar_tokens","unk_tokens","internal_ignored_positions","special_positions",
                            "special_labels_wrong","wordpiece_continuation_tokens","affected_entities","entities_crossing_truncation"):
                    counts[key] += item[key]
                for key in ("uncovered","overcovered","full_uncovered","truncated_chars","repairs","mismatch"):
                    counts[key+"_chars"] += len(item[key])
                    counts[key+"_sentences"] += bool(item[key])
                counts["needs_truncation_sentences"] += item["needs_truncation"]
                counts["exactly_once_coverage_sentences"] += not item["uncovered"] and not item["overcovered"]
                counts["roundtrip_exact_sentences"] += not item["mismatch"]
                counts["multichar_token_sentences"] += item["multichar_tokens"] > 0
                counts["lossy_multichar_tokens"] += len(item["lossy_tokens"])
                counts["lossy_multichar_sentences"] += bool(item["lossy_tokens"])
                counts["internal_ignore_hole_sentences"] += item["internal_ignored_positions"] > 0
                reasons.update(item["loss_reasons"])
                kinds = [key for key in ("uncovered","overcovered","truncated_chars","repairs","mismatch") if item[key]]
                if kinds:
                    issue = {"split":name,"dialogue_id":ex["dialogue_id"],"sentence_id":ex["sentence_id"],
                             "types":kinds,"n_chars":item["n_chars"],
                             "mismatch_positions":item["mismatch"],"uncovered_positions":item["uncovered"],
                             "repair_positions":item["repairs"],"repair_pairs":item["repair_pairs"],
                             "lossy_tokens":item["lossy_tokens"],"loss_reasons":item["loss_reasons"]}
                    all_issues.append(issue)
                    affected_ids.append({"dialogue_id":ex["dialogue_id"],"sentence_id":ex["sentence_id"]})
                    private.write(json.dumps({**issue,"text":ex["text"],"gold":ex["tags"],
                                  "restored":item["restored"],"tokens":tok.convert_ids_to_tokens(enc["input_ids"]),
                                  "offsets":item["offsets"]},ensure_ascii=False)+"\n")
                if (index+1) % 20000 == 0:
                    print("AUDITED",name,index+1,flush=True)
            summaries[name] = {"dialogues":len(split[name]),"counts":dict(counts),
                               "loss_reasons":dict(reasons),"affected_ids":affected_ids,
                               "char_length_distribution":lengths(char_lens),
                               "untruncated_token_length_with_special_distribution":lengths(token_lens)}
            print("SUMMARY",name,json.dumps(dict(counts)),flush=True)
            del loader,examples

    artificial = [
        ("punctuation","孩子发热，吃药。"),
        ("english_digits","ABC38.5mg"),
        ("ascii_spaces_tabs","A B\t中"),
        ("unicode_contract","A 😀，𠀀e\u0301"),
        ("combining_and_precomposed","e\u0301é"),
        ("control_and_nbsp","中\x00文\u00a0A"),
        ("long_entity_crossing_limit","中"*300),
    ]
    probes = []
    for name,text in artificial:
        tags = ["O"]*len(text)
        if name == "long_entity_crossing_limit":
            tags[251] = "B-Symptom"
            tags[252:] = ["I-Symptom"]*(len(text)-252)
        r = inspect(text,tags,tok)
        probes.append({"name":name,"text":text,"gold":tags,"tokens":tok.tokenize(text),**r})
    # Exercise the public repair path with a real tokenizer and deliberately invalid
    # synthetic labels, separately from the valid real-data gold roundtrip counts.
    ids,off = encode_sentence(tok,"发热")
    invalid = [-100 if s==e else LABEL2ID["I-Symptom"] for s,e in off]
    repaired,uncovered = tokens_to_char_labels(invalid,off,2)
    probes.append({"name":"synthetic_invalid_prediction_repair","offsets":off,
                   "token_labels":invalid,"restored":repaired,"uncovered":uncovered})
    dump(EVIDENCE / "synthetic_tokenizer_probes.json",probes)
    dump(EVIDENCE / "batch_checks.json",batches)
    with (EVIDENCE / "alignment_issues.jsonl").open("x",encoding="utf-8") as f:
        for issue in all_issues:
            f.write(json.dumps(issue,ensure_ascii=False)+"\n")
    original_record = json.loads((Path(STATE["interface"]) / "code/artifacts/alignment_report_train.json").read_text(encoding="utf-8"))
    summary = {"started_at":STATE["started_at"],"finished_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "elapsed_seconds":time.perf_counter()-t0,"scope":"Full internal train+val from official train; no dev/test use",
               "max_length":MAX_LEN,"split_version":split["version"],"split_seed":split["seed"],
               "split_sha256":hashlib.sha256(split_path.read_bytes()).hexdigest(),
               "tokenizer_class":type(tok).__name__,"tokenizer_fast":tok.is_fast,
               "tokenizer_backend_normalizer":str(tok.backend_tokenizer.normalizer),
               "versions":{p:metadata.version(p) for p in ("torch","transformers","tokenizers","numpy")},
               "splits":summaries,"issues_count":len(all_issues),
               "prior_report_roundtrip_mismatch_sentences":original_record["roundtrip_mismatch_sentences"],
               "actual_roundtrip_mismatch_sentences":sum(s["counts"]["mismatch_sentences"] for s in summaries.values()),
               "prior_report_has_sample_ids":False,
               "prior_record_comparison_limit":"Can reproduce count and current IDs; original record contains no IDs to prove identity of all 8 prior sentences",
               "real_data_correction_counts_scope":"Gold -> public token projection -> public char restoration, not arbitrary model predictions",
               "private_details_path":str(private_path)}
    dump(EVIDENCE / "alignment_audit.json",summary)
    print("FINISHED",len(all_issues),"issues; report",EVIDENCE / "alignment_audit.json",flush=True)


if __name__ == "__main__":
    main()
