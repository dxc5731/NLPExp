"""Audit delivered v2 against real INTERNAL train/val; no model/mapping implementation.

Records counts, IDs, offsets and labels, never medical text or token strings/IDs.
Synthetic probes explicitly use invented text. Public source is imported unchanged.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
STATE = json.loads((ROOT / '.local/bert_crf/interface_v2_state.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(Path(STATE['interface']) / 'code'))
from src.align import AlignmentError, encode_sentence, load_tokenizer, restore_char_labels, to_legal_bio
from src.config import MAX_LEN
from src.data_io import load_split
from src.dataset import build_loader, collect_examples, collate_fn
from src.labels import LABELS, LABEL2ID
import torch

E = Path(STATE['evidence'])


def dump(name, value):
    with (E / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bio_violations(tags):
    result = []
    for i, tag in enumerate(tags):
        if tag not in LABELS:
            result.append({'position': i, 'reason': 'unknown_label'})
        elif tag.startswith('I-') and (i == 0 or tags[i - 1] not in ('B-' + tag[2:], 'I-' + tag[2:])):
            result.append({'position': i, 'reason': 'illegal_I_predecessor'})
    return result


def spans(tags):
    result = []
    for i, tag in enumerate(tags):
        if tag.startswith('B-'):
            end = i + 1
            while end < len(tags) and tags[end] == 'I-' + tag[2:]:
                end += 1
            result.append((i, end, tag[2:]))
    return result


def distribution(values):
    ordered = sorted(values)
    return {'min': min(values), 'max': max(values), 'mean': sum(values) / len(values),
            'quantiles_nearest_rank': {str(q): ordered[max(0, math.ceil(q * len(ordered)) - 1)] for q in (.5, .9, .95, .99)},
            'histogram': dict(sorted(Counter(values).items()))}


def coverage(offsets, n):
    cov = [0] * n
    invalid = []
    for index, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if not 0 <= s < e <= n:
            invalid.append(index)
            continue
        for k in range(s, e):
            cov[k] += 1
    return cov, invalid


def batch_summary(batch, examples, name):
    # Metadata is compared to actual input but no source text/tensor content emitted.
    for i, ex in enumerate(examples):
        assert batch['text'][i] == ex['text']
        assert (batch['dialogue_id'][i], batch['sentence_id'][i]) == (ex['dialogue_id'], ex['sentence_id'])
    offsets = batch['offsets']
    tokens = (offsets[:, :, 1] > offsets[:, :, 0]) & batch['attention_mask'].bool()
    labels = batch.get('labels')
    lengths = [len(ex['text']) for ex in examples]
    return {'name': name, 'fields': list(batch),
            'tensors': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in batch.items() if isinstance(v, torch.Tensor)},
            'ids': list(zip(batch['dialogue_id'], batch['sentence_id'])),
            'text_preserved': True, 'ids_preserved': True, 'character_lengths_derived': lengths,
            'original_text_sha256': [sha(ex['text'].encode('utf-8')) for ex in examples],
            'attention_lengths': batch['attention_mask'].sum(1).tolist(),
            'nonzero_offset_lengths': tokens.sum(1).tolist(),
            'padding_positions': int((batch['attention_mask'] == 0).sum()),
            'active_zero_width_special_positions': int((batch['attention_mask'].bool() & ~tokens).sum()),
            'labels_minus100': int((labels == -100).sum()) if labels is not None else None,
            'real_offset_minus100_positions': int(((labels == -100) & tokens).sum()) if labels is not None else None,
            'label_mask_first_column': (labels[:, 0] != -100).tolist() if labels is not None else None,
            'attention_equals_label_mask': torch.equal(batch['attention_mask'].bool(), labels != -100) if labels is not None else None,
            'labels_or_character_gold_in_unlabeled_batch': labels is not None,
            'character_gold_in_batch': 'character_labels' in batch or 'char_labels' in batch,
            'explicit_special_tokens_mask_in_batch': 'special_tokens_mask' in batch}


def synthetic_probes(tok):
    cases = [('punctuation', '孩子发热，吃药。'), ('english_digits', 'ABC38.5mg'),
        ('ascii_spaces_tabs', 'A B\t中'), ('unicode_contract', 'A 😀，𠀀e\u0301'),
        ('combining_precomposed', 'e\u0301é'), ('control_nbsp', '中\x00文\u00a0A'),
        ('literal_special_token', '[CLS]中[SEP]'), ('emoji', '中😀文'),
        ('non_BMP', '中𠀀文'), ('combining_cjk', '中\u0301文'), ('only_space', ' '),
        ('empty', ''), ('at_length_limit', '中' * 254), ('long_entity_crossing_limit', '中' * 300)]
    results = []
    for name, text in cases:
        gold = ['O'] * len(text)
        if name == 'long_entity_crossing_limit':
            gold[251] = 'B-Symptom'
            gold[252:] = ['I-Symptom'] * 48
        raw = tok(text, add_special_tokens=True, truncation=False, return_offsets_mapping=True, return_special_tokens_mask=True)
        offsets = [tuple(o) for o in raw['offset_mapping']]
        cov, invalid = coverage(offsets, len(text))
        item = {'artifact_kind': 'synthetic_interface_probe', 'name': name,
                'text': text if len(text) < 30 else None, 'n_char': len(text),
                'n_untruncated_tokens': len(raw['input_ids']), 'offsets': offsets if len(text) < 30 else None,
                'tokenizer_special_tokens_mask': raw['special_tokens_mask'] if len(text) < 30 else None,
                'uncovered_positions': [i for i, v in enumerate(cov) if not v],
                'multiple_coverage_positions': [i for i, v in enumerate(cov) if v > 1], 'invalid_offsets': invalid}
        try:
            enc = encode_sentence(tok, text, tags=gold, where='synthetic/' + name)
            restored = restore_char_labels(enc.token_labels, enc.offsets, len(text))
            item.update(status='accepted', roundtrip_equal=restored == gold, restored=restored if len(text) < 30 else None,
                        returned_tokens=enc.n_tokens, gold_unchanged=gold == ['O'] * len(text))
            assert enc.n_tokens == len(raw['input_ids'])
        except AlignmentError as exc:
            item.update(status='explicitly_rejected', exception_type=type(exc).__name__, message=str(exc))
        results.append(item)
    # Raw restoration vs explicit repair: no model predictions or real gold rewriting.
    restored = restore_char_labels([LABEL2ID['I-Symptom']], [(0, 2)], 2)
    assert restored == ['I-Symptom', 'I-Symptom']
    explicit, count = to_legal_bio(['O', 'I-Symptom', 'I-Symptom', 'O', 'I-Drug'])
    assert count == 2
    missing = restore_char_labels([LABEL2ID['O']], [(0, 2)], 3)
    assert missing == ['O', 'O', None]
    try:
        to_legal_bio(missing)
    except AlignmentError:
        missing_export_rejected = True
    else:
        missing_export_rejected = False
    # Low-level helper robustness, artificial inputs only: not a valid pipeline.
    malformed = [
        ('short_label_list', lambda: restore_char_labels([0], [(0, 1), (1, 2)], 2)),
        ('negative_label_id', lambda: restore_char_labels([-1], [(0, 1)], 1)),
        ('overlapping_offsets', lambda: restore_char_labels([0, 1], [(0, 2), (1, 2)], 2)),
        ('unknown_tag_export', lambda: to_legal_bio(['B-Unknown'])),
        ('illegal_gold_BIO', lambda: encode_sentence(tok, '中', tags=['I-Symptom']).token_labels),
    ]
    malformed_results = []
    for name, operation in malformed:
        try:
            value = operation()
            malformed_results.append({'name': name, 'status': 'accepted', 'returned': value})
        except Exception as exc:
            malformed_results.append({'name': name, 'status': 'rejected', 'exception': type(exc).__name__})
    dump('synthetic_probes.json', {'scope': 'Invented text or deliberately malformed helper input; never model predictions',
        'tokenizer_probes': results, 'raw_illegal_I_unchanged': restored,
        'explicit_repair_only': {'input': ['O', 'I-Symptom', 'I-Symptom', 'O', 'I-Drug'], 'output': explicit, 'count': count},
        'missing_restored': missing, 'missing_export_rejected': missing_export_rejected,
        'helper_validation_gaps': malformed_results})


def main():
    timer = time.perf_counter()
    tokenizer = load_tokenizer()
    assert tokenizer.is_fast
    split_path = Path(STATE['interface']) / 'code/artifacts/split_v1.json'
    split = json.loads(split_path.read_text(encoding='utf-8'))
    data = load_split('train')  # Only official train; never dev/test.
    assert set(split['train']).isdisjoint(split['val'])
    assert set(split['train']) | set(split['val']) == set(data)
    assert len(split['train']) == len(set(split['train'])) and len(split['val']) == len(set(split['val']))
    assert len(data) == 2472
    batches, issues, summaries = [], [], {}
    for name in ('train', 'val'):
        examples = collect_examples(data, split[name])
        expected = [(did, sent) for did in split[name] for sent in data[did]['dialogue']]
        assert len(expected) == len(examples)
        for ex, (did, sent) in zip(examples, expected):
            assert ex == {'dialogue_id': did, 'sentence_id': sent['sentence_id'], 'text': sent['sentence'], 'tags': sent['BIO_label'].split(' ')}
        loader = build_loader(examples, tokenizer, batch_size=4, shuffle=False, num_workers=0)
        ds = loader.dataset
        batch = next(iter(loader))
        batches.append(batch_summary(batch, examples[:4], name + '_first_batch'))
        count = Counter({k: 0 for k in ('gold_illegal_sentences', 'gold_mutation_sentences', 'uncovered_chars', 'overcovered_chars',
            'invalid_offsets', 'actual_truncated_sentences', 'actual_truncated_chars', 'entities_affected_by_truncation',
            'automatic_I_to_B_changes', 'automatic_O_fill_chars', 'concrete_restored_mismatch_chars', 'special_labels_wrong',
            'raw_input_difference_sentences', 'empty_sentences', 'whitespace_characters', 'non_BMP_characters',
            'roundtrip_exact_sentences', 'roundtrip_incomplete_sentences', 'dataset_char_gold_exact_sentences',
            'special_token_mask_vs_offsets_mismatches', 'token_filtered_gold_illegal_sentences', 'internal_label_ignore_hole_sentences')})
        char_lengths, token_lengths, seen = [], [], set()
        ambiguous_rows = []
        for index, (ex, enc, (_, sent)) in enumerate(zip(examples, ds.encoded, expected)):
            key = (ex['dialogue_id'], ex['sentence_id'])
            assert key not in seen
            seen.add(key)
            text, gold, offsets = ex['text'], ex['tags'], enc['offsets']
            n = len(text)
            raw = tokenizer(text, add_special_tokens=True, truncation=False, return_offsets_mapping=True, return_special_tokens_mask=True)
            assert raw['input_ids'] == enc['input_ids'] and [tuple(o) for o in raw['offset_mapping']] == offsets
            assert len(offsets) == len(enc['labels']) == len(enc['input_ids'])
            count['sentences'] += 1
            count['characters'] += n
            count['tokens_including_special'] += len(offsets)
            count['empty_sentences'] += n == 0
            count['whitespace_characters'] += sum(c.isspace() for c in text)
            count['non_BMP_characters'] += sum(ord(c) > 0xFFFF for c in text)
            violations = bio_violations(gold)
            count['gold_illegal_sentences'] += bool(violations)
            assert len(gold) == n
            assert ds.char_labels[index] == gold == sent['BIO_label'].split(' ')
            count['dataset_char_gold_exact_sentences'] += 1
            cov, bad_offsets = coverage(offsets, n)
            missing = [i for i, v in enumerate(cov) if v == 0]
            multiple = [i for i, v in enumerate(cov) if v > 1]
            count['uncovered_chars'] += len(missing)
            count['overcovered_chars'] += len(multiple)
            count['invalid_offsets'] += len(bad_offsets)
            count['exactly_once_coverage_sentences'] += not missing and not multiple and not bad_offsets
            char_lengths.append(n)
            token_lengths.append(len(offsets))
            restored = restore_char_labels(enc['labels'], offsets, n)
            none_positions = [i for i, tag in enumerate(restored) if tag is None]
            concrete_mismatch = [i for i, (a, b) in enumerate(zip(gold, restored)) if b is not None and a != b]
            count['roundtrip_exact_sentences'] += restored == gold
            count['roundtrip_incomplete_sentences'] += bool(none_positions)
            count['restored_None_characters'] += len(none_positions)
            count['concrete_restored_mismatch_chars'] += len(concrete_mismatch)
            count['automatic_I_to_B_changes'] += sum(gold[i].startswith('I-') and restored[i] == 'B-' + gold[i][2:] for i in concrete_mismatch)
            count['automatic_O_fill_chars'] += sum(restored[i] == 'O' for i in missing)
            ambiguous = []
            ignore_between = []
            filtered_tags = []
            for ti, ((s, e), label, special) in enumerate(zip(offsets, enc['labels'], raw['special_tokens_mask'])):
                count['special_token_mask_vs_offsets_mismatches'] += bool(special) != (s == e)
                if s == e:
                    count['special_positions'] += 1
                    count['special_labels_wrong'] += label != -100
                    continue
                count['non_special_tokens'] += 1
                if e - s > 1:
                    count['multichar_tokens'] += 1
                    count['multichar_tokens_different_BIO_tags'] += len(set(gold[s:e])) > 1
                    count['multichar_tokens_entity_start_inside'] += any(t.startswith('B-') for t in gold[s + 1:e])
                    count['multichar_tokens_entity_end_inside'] += any(gold[k - 1] != 'O' and gold[k] == 'O' for k in range(s + 1, e))
                if label == -100:
                    ignore_between.append(ti)
                    intersects = [(a, b, t) for a, b, t in spans(gold) if a < e and b > s]
                    ambiguous.append({'token_index': ti, 'span': [s, e], 'gold_tags': gold[s:e], 'intersecting_gold_entities': intersects})
                else:
                    filtered_tags.append(LABELS[label])
            count['internal_label_ignore_hole_sentences'] += bool(ignore_between)
            count['token_filtered_gold_illegal_sentences'] += bool(bio_violations(filtered_tags))
            if ambiguous:
                ambiguous_rows.append(index)
                count['ambiguous_tokens'] += len(ambiguous)
                count['ambiguous_sentences'] += 1
                count['ambiguous_characters'] += sum(a['span'][1] - a['span'][0] for a in ambiguous)
            if ambiguous or violations or missing or multiple or concrete_mismatch:
                issues.append({'split': name, 'dialogue_id': key[0], 'sentence_id': key[1], 'n_char': n,
                    'ambiguous': ambiguous, 'missing_restored_label_positions': none_positions,
                    'concrete_mismatch_positions': concrete_mismatch, 'offset_uncovered_positions': missing,
                    'offset_multiple_positions': multiple, 'gold_bio_violations': violations,
                    'text_sha256': sha(text.encode('utf-8')), 'source_text_omitted': True})
            if (index + 1) % 25000 == 0:
                print('AUDITED', name, index + 1, flush=True)
        assert len(ds.ambiguous) == count['ambiguous_tokens']
        assert ds.n_ambiguous_sentences == count['ambiguous_sentences']
        if ambiguous_rows:
            rows = ambiguous_rows[:4]
            b = collate_fn([ds[i] for i in rows], tokenizer.pad_token_id)
            batches.append(batch_summary(b, [examples[i] for i in rows], name + '_ambiguous_batch'))
        # Truly label-free records from permitted official train, no BIO field passed.
        selected = examples[:2] + [examples[i] for i in ambiguous_rows[:2]]
        no_labels_data = {}
        for ex in selected:
            no_labels_data.setdefault(ex['dialogue_id'], {'dialogue': []})['dialogue'].append({'sentence_id': ex['sentence_id'], 'sentence': ex['text']})
        inference_examples = collect_examples(no_labels_data, require_labels=False)
        unlabeled = build_loader(inference_examples, tokenizer, batch_size=len(selected), with_labels=False, shuffle=False, num_workers=0)
        ib = next(iter(unlabeled))
        assert 'labels' not in ib and unlabeled.dataset.char_labels is None
        by_id = {(ex['dialogue_id'], ex['sentence_id']): enc for ex, enc in zip(examples, ds.encoded)}
        for ex, encoded in zip(inference_examples, unlabeled.dataset.encoded):
            source = by_id[(ex['dialogue_id'], ex['sentence_id'])]
            assert source['input_ids'] == encoded['input_ids'] and source['offsets'] == encoded['offsets']
        batches.append(batch_summary(ib, inference_examples, name + '_unlabeled_batch'))
        expected_count = {'train': 88842, 'val': 9687}[name]
        assert len(examples) == expected_count
        summaries[name] = {'dialogues': len(split[name]), 'counts': dict(count),
            'char_lengths': distribution(char_lengths), 'token_lengths_with_special': distribution(token_lengths),
            'unlabeled_examples': len(inference_examples), 'unlabeled_inputs_equal_supervised': True}
        print(name, json.dumps(dict(count)), flush=True)
        del loader, ds, examples, expected
    prior = [json.loads(line) for line in (ROOT / 'docs/bert_crf/evidence/20261019T162653+0800_stage1a/alignment_issues.jsonl').read_text(encoding='utf-8').splitlines()]
    old_ids = {(x['dialogue_id'], x['sentence_id']) for x in prior}
    new_ids = {(x['dialogue_id'], x['sentence_id']) for x in issues if x['ambiguous']}
    assert old_ids == new_ids
    synthetic_probes(tokenizer)
    dump('batch_checks.json', batches)
    with (E / 'alignment_issues.jsonl').open('x', encoding='utf-8') as f:
        for issue in issues:
            f.write(json.dumps(issue, ensure_ascii=False) + '\n')
    total = Counter()
    for s in summaries.values():
        total.update(s['counts'])
    dump('alignment_audit.json', {'artifact_kind': 'interface_audit', 'scope': 'Full fixed internal train+val, official train only',
        'started_at': STATE['started_at'], 'finished_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'elapsed_seconds': time.perf_counter() - timer, 'max_length': MAX_LEN, 'model_used': False,
        'tokenizer': {'class': type(tokenizer).__name__, 'is_fast': tokenizer.is_fast, 'normalizer': str(tokenizer.backend_tokenizer.normalizer)},
        'split_version': split['version'], 'split_seed': split['seed'], 'split_source_sha256': split['source_sha256'],
        'splits': summaries, 'total': dict(total), 'issues': len(issues),
        'roundtrip_nonexact_sentence_fraction': (total['sentences'] - total['roundtrip_exact_sentences']) / total['sentences'],
        'missing_label_character_fraction': total['restored_None_characters'] / total['characters'],
        'previous_8_ids_exact_match': old_ids == new_ids,
        'real_data_to_legal_bio_calls': 0,
        'interpretation': 'Offsets cover all characters; 8 ambiguous token golds still cannot roundtrip, now explicit -100 -> None. Original Dataset character gold stays exact for all sentences. No metric/performance calculation.'})
    print('AUDIT_COMPLETE', total['sentences'], total['characters'], 'sentences/chars;', len(issues), 'issue sentences')


if __name__ == '__main__':
    main()
