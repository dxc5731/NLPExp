"""Audit an explicitly executed real-BERT trainer smoke; no implicit reruns/downloads."""
import csv
import json
from pathlib import Path

import pytest

from bert_crf.experiment import file_hash


def test_pretrained_trainer_execution_evidence(request):
    summary = request.config.getoption('--trainer-smoke-summary')
    if not summary:
        pytest.skip('Explicit real-BERT trainer smoke summary required')
    report = json.loads(Path(summary).read_text(encoding='utf-8'))
    assert report['artifact_kind'] == 'synthetic_smoke_test'
    assert report['status'] == 'PASS' and not report['real_data_used']
    assert all(change > 0 for change in report['max_parameter_changes'].values())
    run = Path(report['run_directory'])
    state = report['state']
    assert state['epoch'] == report['config']['epochs']
    assert state['global_step'] == 6
    assert file_hash(state['best_checkpoint']) == state['best_checkpoint_sha256']
    rows = list(csv.DictReader((run / 'history.csv').open(encoding='utf-8')))
    assert len(rows) == report['config']['epochs']
    best = next(row for row in rows if int(row['epoch']) == state['best_epoch'])
    assert float(best['val_entity_f1']) == report['best_reloaded_validation']['entity']['micro']['f1']
    events = [json.loads(line) for line in (run / 'train.log').read_text(encoding='utf-8').splitlines()]
    actual = [e['history'] for e in events if e['event'] == 'epoch_complete']
    for row, event in zip(rows, actual):
        assert all(float(row[key]) == value for key, value in event.items())
