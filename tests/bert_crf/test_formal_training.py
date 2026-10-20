"""Stage 1E: isolated synthetic updates and guarded formal-startup checks.

No test invokes fit on medical data. Real preflight evidence is separately run
with --preflight and contains zero optimizer steps.
"""
import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bert_crf.experiment import ROOT, file_hash
from bert_crf.formal_config import FormalConfig
from bert_crf.preflight import ReadyContext, check_output_paths
from bert_crf.run_artifacts import ValidationArtifacts
from bert_crf.synthetic_smoke import tiny_components
from bert_crf.trainer import Trainer
from bert_crf.training_config import TrainingConfig


@pytest.fixture
def candidate():
    return json.loads((ROOT / 'configs/bert_crf/first_baseline.json').read_text(encoding='utf-8'))


def test_explicit_config_roundtrip(candidate, tmp_path):
    c = FormalConfig(candidate, root=ROOT)
    assert c.training.seed == c.model.seed == 42
    assert c.training.learning_rate == 2e-5
    assert c.training.selection_metric == 'entity.micro.f1'
    assert c.value['alignment']['gold'] == 'original_character_bio'
    original = c.identity
    candidate['training']['learning_rate'] = 123
    assert c.identity == original  # defensive copy
    p = tmp_path / 'config.json'
    p.write_text(json.dumps(c.value), encoding='utf-8')
    assert FormalConfig.load(p, root=ROOT).identity == original


@pytest.mark.parametrize('path,value', [
    (('artifact_kind',), 'synthetic_smoke_test'),
    (('model', 'bert_revision'), 'unfixed'),
    (('model', 'bio_constraints'), False),
    (('training', 'seed'), 1),
    (('training', 'precision'), 'float16'),
    (('training', 'selection_metric'), 'loss'),
    (('training', 'warmup_steps'), 0),
    (('training', 'model_config_path'), 'configs/bert_crf/stage1b_core_smoke.json'),
    (('data', 'source'), 'official_dev'),
    (('data', 'train_split'), 'val'),
    (('alignment', 'gold'), 'token_labels'),
    (('alignment', 'uncovered'), 'fill_O'),
    (('resources', 'gradient_accumulation_steps'), 2),
    (('resources', 'amp'), True),
    (('resources', 'num_workers'), 1),
    (('resources', 'max_tokens'), True),
    (('resources', 'minimum_free_bytes'), 0),
    (('resources', 'padding'), 'truncate'),
    (('resources', 'measurement_sha256'), 'unknown'),
    (('outputs', 'root'), '../other'),
    (('outputs', 'root'), '.local/runs'),
    (('outputs', 'checkpoints'), '../checkpoints'),
    (('inputs', 'official_train', 'files', '.'), '0' * 64),
    (('formats', 'future_handoff'), 'fake'),
    (('executor',), ''),
])
def test_invalid_formal_parameters_rejected(candidate, path, value):
    target = candidate
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises((ValueError, TypeError)):
        FormalConfig(candidate, root=ROOT)


@pytest.mark.parametrize('part', ['model', 'training', 'inputs', 'resources', 'outputs'])
def test_missing_and_unknown_fields_rejected(candidate, part):
    original = candidate[part]
    candidate[part] = {**original, 'unrecorded_override': 1}
    with pytest.raises(ValueError):
        FormalConfig(candidate, root=ROOT)
    candidate[part] = dict(original)
    candidate[part].pop(next(iter(original)))
    with pytest.raises(ValueError):
        FormalConfig(candidate, root=ROOT)


def test_duplicate_json_and_unsafe_manifest_rejected(candidate, tmp_path):
    p = tmp_path / 'duplicate.json'
    p.write_text('{"artifact_kind":"real_training","artifact_kind":"synthetic_smoke_test"}', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate'):
        FormalConfig.load(p, root=ROOT)
    candidate['inputs']['tokenizer']['files']['../outside'] = '0' * 64
    with pytest.raises(ValueError, match='Unsafe'):
        FormalConfig(candidate, root=ROOT)


def test_input_identity_refuses_missing_file(candidate, tmp_path):
    from bert_crf.real_data import verify_inputs
    candidate['inputs']['official_train']['path'] = 'missing/IMCS-V2_train.json'
    with pytest.raises(ValueError, match='identity mismatch'):
        verify_inputs(FormalConfig(candidate, root=tmp_path))


def test_output_write_and_no_dangerous_overwrite(candidate, tmp_path):
    c = FormalConfig(candidate, root=tmp_path)
    c.value['resources']['minimum_disk_free_bytes'] = 1
    check_output_paths(c)
    assert not list((tmp_path / 'experiments/bert_crf').iterdir())
    other = tmp_path / 'file/experiments'
    other.parent.mkdir()
    other.write_text('occupied', encoding='utf-8')
    c = FormalConfig(candidate, root=tmp_path / 'file')
    with pytest.raises(OSError):
        check_output_paths(c)


def test_cli_requires_explicit_action_and_rejects_synthetic(tmp_path):
    from bert_crf.train import main
    with pytest.raises(SystemExit):
        main(['--config', 'not-read-without-action'])
    with pytest.raises(SystemExit):
        main(['--config', 'unused', '--train', '--preflight'])
    with pytest.raises(SystemExit):
        main(['--config', 'unused', '--preflight', '--resume', 'unused'])
    with pytest.raises(ValueError):
        main(['--config', str(ROOT / 'configs/bert_crf/stage1c_trainer_smoke.json'), '--preflight'])


@pytest.mark.parametrize('action', ['--preflight', '--train'])
def test_cli_routes_to_preflight_and_only_explicit_train_dispatches(candidate, monkeypatch, tmp_path, action):
    # Dispatch spy only. No real/synthetic optimizer, real fit or data loaded.
    import bert_crf.train as entry
    config = FormalConfig(candidate, root=ROOT)
    context = SimpleNamespace(model=object(), providers={'train': object(), 'val': object()})
    calls = []
    monkeypatch.setattr(entry.FormalConfig, 'load', lambda *a, **kw: config)
    monkeypatch.setattr(entry, 'run_preflight', lambda *a: calls.append('preflight') or context)
    class Spy:
        def __init__(self, model, training, train, val, root, **kw):
            assert kw['execution_context'] is context
            calls.append('construct')
            self.run = SimpleNamespace(path=tmp_path, event=lambda *a, **kw: None)
        def fit(self, **kw):
            calls.append('fit_dispatch_only')
    monkeypatch.setattr(entry, 'Trainer', Spy)
    entry.main(['--config', 'spy-only', action, '--preflight-output', str(tmp_path / 'check')])
    assert calls == (['preflight'] if action == '--preflight' else ['preflight', 'construct', 'fit_dispatch_only'])


def test_cli_failed_preflight_never_constructs_trainer(candidate, monkeypatch):
    import bert_crf.train as entry
    monkeypatch.setattr(entry.FormalConfig, 'load', lambda *a, **kw: FormalConfig(candidate, root=ROOT))
    def fail(*a):
        raise ValueError('preflight rejected')
    monkeypatch.setattr(entry, 'run_preflight', fail)
    monkeypatch.setattr(entry, 'Trainer', lambda *a, **kw: pytest.fail('Trainer must not be constructed'))
    with pytest.raises(ValueError, match='preflight rejected'):
        entry.main(['--config', 'spy-only', '--train'])


def test_formal_context_binds_exact_model_providers_and_configuration(candidate):
    config = FormalConfig(candidate, root=ROOT)
    model = SimpleNamespace(config=config.model)
    providers = {name: SimpleNamespace(metadata={'artifact_kind': kind, 'formal_config_identity': config.identity})
                 for name, kind in (('train', 'real_training'), ('val', 'real_validation'))}
    context = ReadyContext(config, providers, model, {'status': 'PASS', 'optimizer_steps': 0, 'formal_config_identity': config.identity})
    context.validate_training(model, config.training, providers['train'], providers['val'])
    with pytest.raises(ValueError, match='identity mismatch'):
        context.validate_training(model, config.training, providers['val'], providers['train'])
    with pytest.raises(ValueError, match='config mismatch'):
        context.validate_training(model, replace(config.training, learning_rate=1e-4), providers['train'], providers['val'])
    context.report['status'] = 'FAIL'
    with pytest.raises(RuntimeError):
        context.validate_training(model, config.training, providers['train'], providers['val'])


def make_synthetic(path, name):
    config = replace(TrainingConfig(), device='cpu', learning_rate=0.005, epochs=3)
    model, train, val = tiny_components(config)
    return Trainer(model, config, train, val, path, run_id=name, record_validation_artifacts=True)


def test_synthetic_full_artifact_cycle_and_epoch_resume(tmp_path):
    from test_trainer import assert_nested_equal
    continuous = make_synthetic(tmp_path, 'continuous')
    continuous.fit()
    interrupted = make_synthetic(tmp_path, 'interrupted')
    interrupted.fit(stop_after_epoch=1)
    checkpoint = interrupted.run.path / 'checkpoints/epoch_0001.pt'
    saved = torch.load(checkpoint, weights_only=True)
    assert saved['run_metadata']['run_id'] == 'interrupted'
    assert saved['optimizer']['state'] and saved['scheduler']['last_epoch'] == 2
    resumed = make_synthetic(tmp_path, 'resumed')
    resumed.fit(resume_from=checkpoint)
    assert_nested_equal(continuous.model.state_dict(), resumed.model.state_dict())
    assert_nested_equal(continuous.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_nested_equal(continuous.scheduler.state_dict(), resumed.scheduler.state_dict())
    for trainer in (continuous, resumed):
        p = trainer.run.path
        assert (p / 'predictions.jsonl').exists() and (p / 'errors.jsonl').exists()
        info = json.loads((p / 'run_info.json').read_text(encoding='utf-8'))
        assert info['artifact_kind'] == 'synthetic_smoke_test' and info['official_handoff']['status'] == 'not_run'
        assert info['training_state']['global_step'] == 6
        best = json.loads((p / 'metrics.json').read_text(encoding='utf-8'))
        assert best['best_epoch'] == trainer.state['best_epoch']
        rows = [json.loads(line) for line in (p / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows) == trainer.val_provider.size
        assert all(r['character_count'] == len(r['labels']) and 'text' not in r for r in rows)
        trainer.load_best()
        assert trainer.evaluate()['entity']['micro']['f1'] == trainer.state['best_value']


def test_incomplete_validation_and_checkpoint_failure_mark_failed(tmp_path, monkeypatch):
    trainer = make_synthetic(tmp_path, 'incomplete')
    old = trainer.model.forward
    def bad_decode(**kwargs):
        result = old(**kwargs)
        if kwargs.get('decode'):
            result['paths'][0] = [2] * len(result['paths'][0])  # illegal I start, never repaired
        return result
    trainer.model.forward = bad_decode
    with pytest.raises(ValueError, match='Illegal BIO'):
        trainer.fit()
    p = trainer.run.path
    assert json.loads((p / 'run_info.json').read_text(encoding='utf-8'))['status'] == 'failed'
    assert (p / 'validation/epoch_0001/failure.json').exists()
    assert not (p / 'predictions.jsonl').exists()
    trainer = make_synthetic(tmp_path, 'checkpoint_failed')
    import bert_crf.trainer as module
    def broken(*args, **kwargs):
        raise OSError('Synthetic injected checkpoint write failure')
    monkeypatch.setattr(module, 'save_checkpoint', broken)
    with pytest.raises(OSError, match='write failure'):
        trainer.fit()
    assert json.loads((trainer.run.path / 'run_manifest.json').read_text(encoding='utf-8'))['status'] == 'failed'


def test_validation_artifact_strict_boundaries_and_no_text(tmp_path):
    writer = ValidationArtifacts(tmp_path, 1, 'synthetic_smoke_test')
    writer.add(('synthetic_dialogue', '1'), [1, 2, 0], [1, 0, 0])
    report = writer.finish({'entity': {'micro': {'f1': 0}}}, 0.01)
    errors = json.loads((writer.path / 'errors.jsonl').read_text(encoding='utf-8'))
    assert errors['missed_entities'] == [[0, 2, 'Symptom']]
    assert errors['spurious_entities'] == [[0, 1, 'Symptom']]
    assert report['sentences'] == 1


def test_real_preflight_and_resource_evidence_if_explicit():
    import os
    state_path = os.environ.get('STAGE1E_STATE')
    if not state_path:
        pytest.skip('Explicit real-data preflight evidence required')
    state = json.loads(Path(state_path).read_text(encoding='utf-8'))
    directory = Path(state['evidence']) / 'preflight_final'
    report = json.loads((directory / 'preflight.json').read_text(encoding='utf-8'))
    assert report['status'] == 'PASS' and report['optimizer_steps'] == 0
    assert all(c['status'] == 'PASS' for c in report['checks'])
    probes = json.loads((directory / 'real_probes.json').read_text(encoding='utf-8'))
    assert probes['optimizer_steps'] == 0 and not probes['performance_metrics_computed']
    for case in (probes['train'], probes['validation']):
        assert case['parameter_hash_before'] == case['parameter_hash_after']
        assert case['parameter_versions_unchanged']
    lengths = json.loads((directory / 'lengths_and_coverage.json').read_text(encoding='utf-8'))
    assert sum(v['sentences'] for v in lengths.values()) == 98529
    assert sum(v['characters'] for v in lengths.values()) == 1279361


def test_preflight_failure_record_is_not_completed(candidate, tmp_path, monkeypatch):
    import bert_crf.preflight as module
    config = FormalConfig(candidate, root=tmp_path)
    monkeypatch.setattr(module, 'build_providers', lambda c: (_ for _ in ()).throw(ValueError('Synthetic input identity failure')))
    report_dir = tmp_path / 'check'
    with pytest.raises(ValueError, match='identity failure'):
        module.run_preflight(config, report_dir)
    report = json.loads((report_dir / 'preflight.json').read_text(encoding='utf-8'))
    assert report['status'] == 'FAIL' and report['optimizer_steps'] == 0
    assert not list((tmp_path / 'experiments/bert_crf').iterdir())
    with pytest.raises(FileExistsError):
        module.run_preflight(config, report_dir)


def test_preflight_output_cannot_pollute_inputs_or_formal_runs(candidate):
    from bert_crf.preflight import run_preflight
    config = FormalConfig(candidate, root=ROOT)
    for p in (config.path(candidate['inputs']['tokenizer']['path']) / 'never_create',
              ROOT / 'experiments/bert_crf/never_create', ROOT / 'src/never_create'):
        with pytest.raises(ValueError, match='overlaps'):
            run_preflight(config, p)
        assert not p.exists()


def test_synthetic_pretrained_stage1e_execution_evidence():
    import os
    if not os.environ.get('STAGE1E_STATE'):
        pytest.skip('Explicit Stage 1E synthetic pretrained run required')
    state = json.loads(Path(os.environ['STAGE1E_STATE']).read_text(encoding='utf-8'))
    result = json.loads((Path(state['evidence']) / 'synthetic_pretrained_trainer.json').read_text(encoding='utf-8'))
    assert result['status'] == 'PASS' and result['artifact_kind'] == 'synthetic_smoke_test'
    assert result['real_data_used'] is False and result['state']['global_step'] == 4
    assert all(value > 0 for value in result['max_parameter_changes'].values())
    p = Path(result['run_directory'])
    info = json.loads((p / 'run_info.json').read_text(encoding='utf-8'))
    assert info['status'] == 'completed' and info['ended_at']
    assert info['selected_validation']['metrics']['entity']['micro']['f1'] == result['state']['best_value']


def test_project_launcher_has_no_default_training_action():
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'scripts/train_bert_crf.py'),
                             '--config', 'not_read_without_action'], capture_output=True, text=True, encoding='utf-8', env=env)
    assert result.returncode == 2
    assert 'one of the arguments --preflight --train is required' in result.stderr


@pytest.mark.parametrize('case', ['measurement_hash', 'gpu_margin'])
def test_critical_resource_failures_block_startup(candidate, tmp_path, monkeypatch, case):
    import bert_crf.preflight as module
    measurement = Path(candidate['resources']['measurement_file'])
    (tmp_path / measurement).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / measurement).write_bytes((ROOT / measurement).read_bytes())
    config = FormalConfig(candidate, root=tmp_path)
    if case == 'measurement_hash':
        config.value['resources']['measurement_sha256'] = '0' * 64
    monkeypatch.setattr(module, 'build_providers', lambda c: ({}, None))
    monkeypatch.setattr(module, 'scan_providers', lambda c, p: {'train': {'sentences': 88842, 'characters': 1153312,
                                                                         'token_lengths': {'max': 201}}})
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda *a: (1, 8 * 1024**3))
    monkeypatch.setattr(module, 'build_model', lambda c: pytest.fail('Resource failures must precede model/optimizer'))
    directory = tmp_path / case
    with pytest.raises((ValueError, RuntimeError), match='measurement hash|free memory'):
        module.run_preflight(config, directory)
    assert json.loads((directory / 'preflight.json').read_text(encoding='utf-8'))['status'] == 'FAIL'


@pytest.fixture(scope='module')
def real_formal_providers():
    import os
    if not os.environ.get('STAGE1E_STATE'):
        pytest.skip('Explicit real-data environment required')
    from bert_crf.real_data import build_providers
    config = FormalConfig.load(ROOT / 'configs/bert_crf/first_baseline.json', root=ROOT)
    providers, _ = build_providers(config)
    return config, providers


def test_formal_real_provider_identity_and_gold(real_formal_providers):
    config, providers = real_formal_providers
    for name, kind in (('train', 'real_training'), ('val', 'real_validation')):
        provider = providers[name]
        assert provider.metadata['artifact_kind'] == kind
        assert provider.metadata['formal_config_identity'] == config.identity
        assert provider.metadata['mapping_policy'] == 'offset_share_unique_v1'
        batch = next(provider.batches(0, 4, False, config.training.seed))
        for row, key in enumerate(batch['ids']):
            text, gold, _, _ = provider.adapter.records[key]
            assert batch['inputs']['character_labels'][row, :len(text)].tolist() == list(gold)
            assert int(batch['inputs']['character_mask'][row].sum()) == len(text)
    assert not set(providers['train'].adapter.records) & set(providers['val'].adapter.records)


def test_real_provider_epoch_order_and_global_rng(real_formal_providers):
    config, providers = real_formal_providers
    provider = providers['train']
    before = torch.get_rng_state().clone()
    first = next(provider.batches(1, 16, True, 42))
    second = next(provider.batches(1, 16, True, 42))
    third = next(provider.batches(2, 16, True, 42))
    assert torch.equal(before, torch.get_rng_state())
    assert first['ids'] == second['ids'] and first['ids'] != third['ids']
    assert torch.equal(first['inputs']['character_labels'], second['inputs']['character_labels'])
