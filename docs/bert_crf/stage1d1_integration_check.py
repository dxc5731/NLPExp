"""Re-runnable real internal-data checks; no optimizer, metrics, export or dev/test.

Use stage1d1_runner.py for environment and append-only command evidence. Outputs
contain IDs, hashes, sizes and diagnostics, never medical text/token IDs/paths.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from bert_crf.character_mapping import MAPPING_POLICY, OFFSET_UNIT, OffsetCharacterAdapter, character_index
from bert_crf.config import ModelConfig, configure_runtime
from bert_crf.constraints import LABELS
from bert_crf.data_adapter import V2DataAdapter, V2BatchProvider, gold_ids
from bert_crf.experiment import environment, file_hash, write_json
from bert_crf.model import BertEncoder, CharacterBertCRF
from bert_crf.trainer import move

ROOT = Path(__file__).resolve().parents[2]
STATE = json.loads(Path(os.environ['STAGE1D1_STATE']).read_text(encoding='utf-8'))
EV = Path(STATE['evidence'])
LOCAL = Path(STATE['local'])
CFG_PATH = ROOT / 'configs/bert_crf/stage1d1_integration.json'
CFG = json.loads(CFG_PATH.read_text(encoding='utf-8'))
assert CFG['optimizer_steps'] == 0 and not CFG['compute_real_f1'] and not CFG['write_prediction_files']
assert CFG['mapping_policy'] == MAPPING_POLICY and CFG['offset_unit'] == OFFSET_UNIT
assert CFG['official_splits_allowed'] == ['train']


def output(name, value):
    path = EV / name
    if path.exists():
        raise FileExistsError(path)
    write_json(path, value)


def resources():
    # Actual A package, untouched; isolated sys.path comes from the runner.
    from src.align import load_tokenizer
    from src.data_io import load_split
    from src.dataset import build_loader, collect_examples, collate_fn
    interface = Path(STATE['interface'])
    split_path = interface / 'code/artifacts/split_v1.json'
    split = json.loads(split_path.read_text(encoding='utf-8'))
    data = load_split('train')  # Sole official split ever read in this script.
    assert not set(split['train']) & set(split['val'])
    assert set(split['train']) | set(split['val']) == set(data)
    tok = load_tokenizer()
    assert tok.is_fast
    return data, split, split_path, tok, collect_examples, build_loader, collate_fn


def describe(batch):
    tensors = ('input_ids', 'attention_mask', 'offsets', 'character_labels', 'character_mask',
               'character_lengths', 'special_tokens_mask')
    return {'tensors': {k: None if getattr(batch, k) is None else {
            'shape': list(getattr(batch, k).shape), 'dtype': str(getattr(batch, k).dtype)} for k in tensors},
            'ids': batch.ids, 'character_lengths': batch.character_lengths.tolist(),
            'text_sha256': [hashlib.sha256(t.encode('utf-8')).hexdigest() for t in batch.texts],
            'gold_source': 'Dataset.char_labels via verified composite ID; token labels unused',
            'metadata_fields': ['dialogue_ids', 'sentence_ids', 'texts'], 'medical_text_omitted': True}


def scan():
    tick = time.perf_counter()
    output('environment.json', environment())
    output('integration_config.json', CFG)
    data, split, split_path, tok, collect, build, collate = resources()
    previous_identity = json.loads((EV / 'input_identity.json').read_text(encoding='utf-8'))
    assert file_hash(split_path) == previous_identity[str(split_path)]
    raw = {(d, s['sentence_id']): s for d, dialog in data.items() for s in dialog['dialogue']}
    assert len(raw) == sum(len(d['dialogue']) for d in data.values())
    reports, first, issues = {}, {}, []
    required = [json.loads(line) for line in Path(STATE['prior_issues']).read_text(encoding='utf-8').splitlines()]
    eight = []
    for name in ('train', 'val'):
        examples = collect(data, split[name])
        loader = build(examples, tok, batch_size=CFG['audit_batch_size'], num_workers=CFG['num_workers'])
        # Validate source lineage independent of the Dataset. No tag restoration.
        for ex, tags in zip(loader.dataset.examples, loader.dataset.char_labels):
            source = raw[(ex['dialogue_id'], ex['sentence_id'])]
            assert ex['text'] == source['sentence']
            assert ex['tags'] == tags == source['BIO_label'].split(' ')
        adapter = V2DataAdapter(loader.dataset, tok)
        report = Counter(dialogues=len(split[name]), sentences=len(examples), characters=sum(len(x['text']) for x in examples))
        for field in ('accepted_sentences', 'characters_covered_exactly_once', 'coverage_gap_characters',
                      'coverage_overlap_characters', 'invalid_offset_sentences', 'gold_length_mismatch_sentences',
                      'illegal_gold_bio_sentences', 'rejected_sentences', 'gold_unchanged_sentences',
                      'multichar_tokens', 'unlabeled_records_checked'):
            report[field] = 0
        seen = set()
        for batch in loader:
            try:
                converted = adapter.adapt(batch)
            except (ValueError, KeyError) as exc:
                # Never skip: identify rejected IDs and fail the final acceptance.
                for d, s in zip(batch['dialogue_id'], batch['sentence_id']):
                    index = next(i for i, ex in enumerate(examples) if (ex['dialogue_id'], ex['sentence_id']) == (d, s))
                    try:
                        adapter.adapt(collate([loader.dataset[index]], tok.pad_token_id))
                    except (ValueError, KeyError) as error:
                        reason = getattr(error, 'reason', type(error).__name__)
                        issues.append({'split': name, 'id': [d, s], 'reason': reason,
                            'positions': getattr(error, 'positions', []), 'text_omitted': True})
                        report['rejected_sentences'] += 1
                raise RuntimeError(f'Unresolved real-data adapter failure in {name}; see diagnostic context') from exc
            if name not in first:
                first[name] = describe(converted)
            index, mask = character_index(**converted.model_inputs()['mapping'])
            assert torch.equal(mask, converted.character_mask)
            for row, key in enumerate(converted.ids):
                assert key not in seen
                seen.add(key)
                n = int(converted.character_lengths[row])
                source = raw[key]
                assert converted.character_labels[row, :n].tolist() == list(gold_ids(source['BIO_label'].split(' '), source['sentence']))
                for c in range(n):
                    j = int(index[row, c])
                    start, end = converted.offsets[row, j].tolist()
                    assert start <= c < end and bool(converted.attention_mask[row, j]) and not bool(converted.special_tokens_mask[row, j])
                report['accepted_sentences'] += 1
                report['gold_unchanged_sentences'] += 1
                report['characters_covered_exactly_once'] += n
            spans = converted.offsets
            report['multichar_tokens'] += int(((spans[:, :, 1] - spans[:, :, 0]) > 1).sum())
        assert len(seen) == len(examples)
        # Required eight IDs originate from D0's actual anomaly file, not a new list.
        by_id = {(ex['dialogue_id'], ex['sentence_id']): i for i, ex in enumerate(examples)}
        for issue in [x for x in required if x['split'] == name]:
            key = (issue['dialogue_id'], issue['sentence_id'])
            i = by_id[key]
            batch = adapter.adapt(collate([loader.dataset[i]], tok.pad_token_id))
            assert hashlib.sha256(batch.texts[0].encode('utf-8')).hexdigest() == issue['text_sha256']
            assert batch.character_lengths.tolist() == [issue['n_char']]
            indices, _ = character_index(**batch.model_inputs()['mapping'])
            for ambiguous in issue['ambiguous']:
                start, end = ambiguous['span']
                assert len(set(indices[0, start:end].tolist())) == 1
                assert [LABELS[x] for x in batch.character_labels[0, start:end].tolist()] == ambiguous['gold_tags']
            eight.append({'split': name, 'id': list(key), 'n_char': issue['n_char'], 'text_sha256': issue['text_sha256'],
                'source_gold_sha256': hashlib.sha256(raw[key]['BIO_label'].encode('utf-8')).hexdigest(),
                'raw_gold_unchanged': True, 'character_gold_has_ignore_index': False,
                'unique_full_coverage': True, 'multichar_shared_source_checked': True,
                'ambiguous_spans': [a['span'] for a in issue['ambiguous']], 'source_text_omitted': True})
        # A label-free Dataset assembled WITHOUT tags, including all ambiguous IDs.
        chosen = [by_id[tuple(x['id'])] for x in eight if x['split'] == name]
        chosen = list(dict.fromkeys(chosen + [0, 1]))
        infer_examples = [{k: ex[k] for k in ('dialogue_id', 'sentence_id', 'text')} for i in chosen for ex in [examples[i]]]
        infer_loader = build(infer_examples, tok, batch_size=16, with_labels=False, num_workers=0)
        infer_adapter = V2DataAdapter(infer_loader.dataset, tok)
        for raw_infer in infer_loader:
            assert 'labels' not in raw_infer
            unlabeled = infer_adapter.adapt(raw_infer)
            assert unlabeled.character_labels is None
            for row, key in enumerate(unlabeled.ids):
                labeled = adapter.adapt(collate([loader.dataset[by_id[key]]], tok.pad_token_id))
                n = len(labeled.texts[0])
                assert torch.equal(unlabeled.character_mask[row, :n], labeled.character_mask[0])
                assert torch.equal(unlabeled.offsets[row, :labeled.input_ids.shape[1]], labeled.offsets[0])
            report['unlabeled_records_checked'] += len(unlabeled.ids)
        # Provider reuses public collate and cannot start Stage1C real training.
        provider = V2BatchProvider(loader, tok, provenance={'artifact_kind': 'real_data_integration_test',
            'data_sha256': file_hash(Path(STATE['data']) / 'IMCS-V2_train.json'),
            'split_sha256': file_hash(split_path), 'internal_split': name,
            'interface_zip_sha256': file_hash(STATE['archive']),
            'bert_revision': CFG['bert_revision'], 'tokenizer_provenance': STATE['tokenizer']})
        first_provider = next(provider.batches(0, 2, False, CFG['seed']))
        assert first_provider['ids'] == list(seen)[:0] + [(ex['dialogue_id'], ex['sentence_id']) for ex in examples[:2]]
        first[name]['provider_metadata'] = provider.metadata
        reports[name] = dict(report)
        print(json.dumps({'split_complete': name, **reports[name]}), flush=True)
    assert len(eight) == 8
    total = {key: sum(r[key] for r in reports.values()) for key in reports['train']}
    output('coverage_audit.json', {'artifact_kind': 'real_data_integration_test', 'status': 'PASS',
        'splits': reports, 'total': total, 'issues': issues, 'rejection_reasons': {},
        'source': 'official train ONLY, fixed internal train/val', 'offset_unit': OFFSET_UNIT,
        'mapping_policy': MAPPING_POLICY, 'full_bert_forward': False, 'optimizer_steps': 0,
        'elapsed_seconds': time.perf_counter() - tick})
    output('batch_contract.json', first)
    output('eight_cases_data.json', eight)


def model_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_summary(module):
    result = {'trainable_tensors': 0, 'finite_gradient_tensors': 0, 'nonzero_gradient_tensors': 0, 'missing': [], 'nonfinite': []}
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        result['trainable_tensors'] += 1
        if p.grad is None:
            result['missing'].append(name)
        elif not bool(torch.isfinite(p.grad).all()):
            result['nonfinite'].append(name)
        else:
            result['finite_gradient_tensors'] += 1
            result['nonzero_gradient_tensors'] += int(bool((p.grad != 0).any()))
    assert not result['missing'] and not result['nonfinite'] and result['nonzero_gradient_tensors'] > 0
    return result


def cuda_check():
    tick = time.perf_counter()
    assert json.loads((EV / 'coverage_audit.json').read_text(encoding='utf-8'))['status'] == 'PASS'
    data, split, split_path, tok, collect, build, collate = resources()
    # Tiny actual public loaders, actual fixed split order, not artificial labels.
    actual = {}
    cases = [json.loads(line) for line in Path(STATE['prior_issues']).read_text(encoding='utf-8').splitlines()]
    for name in ('train', 'val'):
        examples = collect(data, split[name])
        wanted = {(x['dialogue_id'], x['sentence_id']) for x in cases if x['split'] == name}
        chosen = examples[:CFG['integration_batch_size']] + [ex for ex in examples if (ex['dialogue_id'], ex['sentence_id']) in wanted]
        loader = build(chosen, tok, batch_size=CFG['integration_batch_size'], num_workers=0)
        adapter = V2DataAdapter(loader.dataset, tok)
        actual[name] = (adapter, loader, collate)
    config = ModelConfig(artifact_kind='real_data_integration_test', bert_model_name=CFG['bert_model_name'],
        bert_revision=CFG['bert_revision'], device=CFG['device'], precision=CFG['precision'],
        seed=CFG['seed'], dropout=CFG['dropout'])
    device = configure_runtime(config)
    assert device.type == 'cuda'
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = CharacterBertCRF(config, BertEncoder.from_pretrained(config, STATE['pretrained']), OffsetCharacterAdapter()).to(device)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    before = model_hash(model)
    versions = {name: p._version for name, p in model.named_parameters()}
    results = {}
    train_adapter, train_loader, _ = actual['train']
    batch = train_adapter.adapt(next(iter(train_loader)))
    inputs = move(batch.model_inputs(), device)
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = model(**inputs, decode=True)
    assert bool(torch.isfinite(out['loss']))
    out['loss'].backward()
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - start
    gradients = {name: gradient_summary(module) for name, module in (
        ('bert', model.encoder), ('linear', model.head.classifier), ('crf', model.head.crf))}
    results['train'] = {**describe(batch), 'loss': float(out['loss'].detach()),
        'forward_backward_seconds': backward_seconds, 'gradients': gradients,
        'prediction_lengths': list(map(len, out['paths'])), 'optimizer_created': False, 'optimizer_steps': 0}
    del out
    model.zero_grad(set_to_none=True)
    model.eval()
    val_adapter, val_loader, _ = actual['val']
    val_batch = val_adapter.adapt(next(iter(val_loader)))
    val_inputs = move(val_batch.model_inputs(), device)
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(**val_inputs, decode=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    for text, path in zip(val_batch.texts, out['paths']):
        assert len(path) == len(text)
        gold_ids([LABELS[x] for x in path], text)
    results['val'] = {**describe(val_batch), 'loss': float(out['loss']), 'forward_decode_seconds': elapsed,
        'prediction_lengths': list(map(len, out['paths'])), 'strict_bio_legal': True,
        'performance_metrics_computed': False, 'model_selection': False}
    before_emissions = out['emissions'].detach().cpu().clone()
    before_paths = out['paths']
    del out
    # Check every anomaly with actual pretrained representations and unchanged gold.
    eight_results = []
    for issue in cases:
        key = (issue['dialogue_id'], issue['sentence_id'])
        adapter, loader, collate = actual[issue['split']]
        indices = [i for i, ex in enumerate(loader.dataset.examples) if (ex['dialogue_id'], ex['sentence_id']) == key]
        assert len(indices) == 1
        converted = adapter.adapt(collate([loader.dataset[indices[0]]], tok.pad_token_id))
        original = next(s for s in data[key[0]]['dialogue'] if s['sentence_id'] == key[1])
        assert converted.character_labels[0].tolist() == list(gold_ids(original['BIO_label'].split(' '), original['sentence']))
        with torch.no_grad():
            result = model(**move(converted.model_inputs(), device), decode=True)
        assert bool(torch.isfinite(result['loss'])) and len(result['paths'][0]) == len(original['sentence'])
        gold_ids([LABELS[i] for i in result['paths'][0]], original['sentence'])
        eight_results.append({'split': issue['split'], 'id': list(key), 'n_char': issue['n_char'],
            'loss': float(result['loss']), 'loss_finite': True, 'raw_gold_unchanged': True,
            'prediction_length': len(result['paths'][0]), 'strict_bio_legal': True})
    # Label-free decode and same mapping; no official inference/export.
    infer_inputs = dict(val_inputs, character_labels=None)
    with torch.no_grad():
        infer = model(**infer_inputs, decode=True)
    assert infer['loss'] is None and infer['paths'] == before_paths
    after = model_hash(model)
    assert before == after and versions == {name: p._version for name, p in model.named_parameters()}
    peak = {'max_allocated_bytes': torch.cuda.max_memory_allocated(), 'max_reserved_bytes': torch.cuda.max_memory_reserved()}
    # Same real adapter / pretrained model checkpoint round-trip, local ignored only.
    checkpoint = LOCAL / 'integrated_checkpoint'
    model.save_pretrained(checkpoint)
    del model, result, infer, inputs, val_inputs
    torch.cuda.empty_cache()
    reload_start = time.perf_counter()
    reloaded = CharacterBertCRF.from_pretrained(checkpoint, adapter=OffsetCharacterAdapter()).to(device).eval()
    with torch.no_grad():
        restored = reloaded(**move(val_batch.model_inputs(), device), decode=True)
    assert torch.equal(before_emissions, restored['emissions'].cpu()) and restored['paths'] == before_paths
    assert model_hash(reloaded) == before
    torch.cuda.synchronize()
    reload_seconds = time.perf_counter() - reload_start
    output('eight_cases_model.json', eight_results)
    output('real_batch_integration.json', {'artifact_kind': 'real_data_integration_test', 'status': 'PASS',
        'model_config': asdict(config), 'parameters': sum(p.numel() for p in reloaded.parameters()),
        'pretrained_path': STATE['pretrained'], 'pretrained_weights_sha256': file_hash(Path(STATE['pretrained']) / 'model.safetensors'),
        'pretrained_load_seconds': load_seconds, 'batches': results,
        'parameter_hash_before': before, 'parameter_hash_after': after,
        'parameter_version_counters_unchanged': True, 'optimizer_created': False, 'optimizer_steps': 0,
        'gold_free_decode_identical': True, 'mapping_policy': MAPPING_POLICY, 'memory_before_reload': peak,
        'checkpoint': {'path': str(checkpoint), 'same_policy': MAPPING_POLICY, 'parameter_hash_equal': True,
                       'eval_emissions_bitwise_equal': True, 'paths_equal': True, 'reload_seconds': reload_seconds},
        'elapsed_seconds': time.perf_counter() - tick, 'performance_metrics_computed': False})
    print(json.dumps({'status': 'PASS', 'train_loss': results['train']['loss'], 'val_loss': results['val']['loss'],
        'eight_cases': len(eight_results), 'optimizer_steps': 0, 'peak_memory': peak}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['scan', 'cuda'])
    args = parser.parse_args()
    try:
        scan() if args.action == 'scan' else cuda_check()
    except Exception as exc:
        # Real failures remain distinguishable from PASS. No medical text.
        write_json(LOCAL / (args.action + '_failure.json'), {'status': 'FAIL', 'exception': type(exc).__name__,
            'reason': getattr(exc, 'reason', str(exc)), 'time': time.time()})
        raise
