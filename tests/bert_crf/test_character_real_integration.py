"""Opt-in real internal-data acceptance; requires the explicit Stage 1D-1 run.

Only the local official train JSON and its fixed train/val IDs are accessed.
The GPU evidence is produced by an explicit no-optimizer script, not fabricated
by these assertions. Ordinary core tests remain independent of medical data.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch

from bert_crf.character_mapping import character_index
from bert_crf.constraints import LABELS
from bert_crf.data_adapter import V2DataAdapter
from bert_crf.experiment import file_hash


@pytest.fixture(scope='module')
def real_context():
    location = os.environ.get('STAGE1D1_STATE')
    if not location:
        pytest.skip('Explicit STAGE1D1_STATE required; no implicit dataset access')
    state = json.loads(Path(location).read_text(encoding='utf-8'))
    from src.align import load_tokenizer
    from src.data_io import load_split
    from src.dataset import build_loader, collect_examples
    data = load_split('train')
    issues = [json.loads(line) for line in Path(state['prior_issues']).read_text(encoding='utf-8').splitlines()]
    keys = {(x['dialogue_id'], x['sentence_id']) for x in issues}
    examples = [ex for ex in collect_examples(data) if (ex['dialogue_id'], ex['sentence_id']) in keys]
    tok = load_tokenizer()
    loader = build_loader(examples, tok, batch_size=1, num_workers=0)
    adapter = V2DataAdapter(loader.dataset, tok)
    batches = {batch.ids[0]: batch for raw in loader for batch in [adapter.adapt(raw)]}
    return state, issues, batches


@pytest.mark.parametrize('case', range(8))
def test_real_eight_gold_and_unique_multichar_sources(real_context, case):
    state, issues, batches = real_context
    issue = issues[case]
    key = (issue['dialogue_id'], issue['sentence_id'])
    batch = batches[key]
    assert hashlib.sha256(batch.texts[0].encode('utf-8')).hexdigest() == issue['text_sha256']
    assert len(batch.texts[0]) == issue['n_char'] == batch.character_labels.shape[1]
    assert bool(batch.character_mask.all()) and not bool((batch.character_labels < 0).any())
    index, _ = character_index(**batch.model_inputs()['mapping'])
    for a in issue['ambiguous']:
        start, end = a['span']
        assert len(set(index[0, start:end].tolist())) == 1
        assert [LABELS[i] for i in batch.character_labels[0, start:end]] == a['gold_tags']
    evidence = json.loads((Path(state['evidence']) / 'eight_cases_model.json').read_text(encoding='utf-8'))
    actual = next(x for x in evidence if tuple(x['id']) == key)
    assert math.isfinite(actual['loss']) and actual['raw_gold_unchanged']
    assert actual['strict_bio_legal'] and actual['prediction_length'] == issue['n_char']


def test_real_coverage_and_gpu_no_update_evidence(real_context):
    state, issues, batches = real_context
    ev = Path(state['evidence'])
    coverage = json.loads((ev / 'coverage_audit.json').read_text(encoding='utf-8'))
    assert coverage['status'] == 'PASS' and not coverage['issues'] and not coverage['rejection_reasons']
    total = coverage['total']
    assert total['sentences'] == total['accepted_sentences'] == total['gold_unchanged_sentences'] == 98529
    assert total['characters'] == total['characters_covered_exactly_once'] == 1279361
    assert all(total[k] == 0 for k in ('coverage_gap_characters', 'coverage_overlap_characters',
        'invalid_offset_sentences', 'gold_length_mismatch_sentences', 'illegal_gold_bio_sentences', 'rejected_sentences'))
    model = json.loads((ev / 'real_batch_integration.json').read_text(encoding='utf-8'))
    assert model['status'] == 'PASS' and model['artifact_kind'] == 'real_data_integration_test'
    assert not model['optimizer_created'] and model['optimizer_steps'] == 0
    assert model['parameter_hash_before'] == model['parameter_hash_after']
    assert model['parameter_version_counters_unchanged'] and model['gold_free_decode_identical']
    assert model['checkpoint']['eval_emissions_bitwise_equal'] and model['checkpoint']['paths_equal']
    assert model['pretrained_weights_sha256'] == file_hash(Path(state['pretrained']) / 'model.safetensors')
    for component in ('bert', 'linear', 'crf'):
        gradient = model['batches']['train']['gradients'][component]
        assert not gradient['missing'] and not gradient['nonfinite']
        assert gradient['trainable_tensors'] == gradient['finite_gradient_tensors']
        assert gradient['nonzero_gradient_tensors'] > 0
    for batch in model['batches'].values():
        assert batch['prediction_lengths'] == batch['character_lengths'] and math.isfinite(batch['loss'])
