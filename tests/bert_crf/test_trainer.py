import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from bert_crf.checkpoint import read_checkpoint, rng_state, restore_rng
from bert_crf.experiment import Experiment, summarize_real_runs
from bert_crf.synthetic_smoke import tiny_components
from bert_crf.trainer import Trainer, metric_value
from bert_crf.training_config import TrainingConfig, require_synthetic


def configuration(**kwargs):
    return replace(TrainingConfig(), **{'device': 'cpu', 'learning_rate': 0.005, **kwargs})


def make(tmp_path, name, config=None):
    config = config or configuration()
    model, train, val = tiny_components(config)
    return Trainer(model, config, train, val, tmp_path, run_id=name)


def json_file(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): assert_nested_equal(a, b)
    else:
        assert left == right


def test_actual_updates_zero_grad_modes_scheduler_and_log_consistency(tmp_path):
    trainer = make(tmp_path, 'updates')
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    modes, clear = [], []
    def observe(module, args):
        modes.append((module.training, torch.is_grad_enabled()))
        if module.training:
            clear.append(all(p.grad is None for p in module.parameters()))
    trainer.model.register_forward_pre_hook(observe)
    trainer.fit()
    assert all(clear) and len(clear) == 6
    assert (True, True) in modes and (False, False) in modes
    assert trainer.model.training
    assert trainer.state['epoch'] == 3 and trainer.state['global_step'] == 6
    assert trainer.scheduler.last_epoch == 6
    assert trainer.optimizer.param_groups[0]['lr'] == 0
    for name in ['encoder.bert.embeddings.word_embeddings.weight', 'head.classifier.weight', 'head.crf.transitions']:
        assert not torch.equal(before[name], trainer.model.state_dict()[name])
    events = [json.loads(line) for line in (trainer.run.path / 'train.log').read_text(encoding='utf-8').splitlines()]
    assert all(e['artifact_kind'] == 'synthetic_smoke_test' for e in events)
    rows = list(csv.DictReader((trainer.run.path / 'history.csv').open(encoding='utf-8')))
    epochs = [e for e in events if e['event'] == 'epoch_complete']
    assert len(rows) == len(epochs) == 3
    for row, event in zip(rows, epochs):
        for key, value in event['history'].items(): assert float(row[key]) == value
    batch = [e for e in events if e['event'] == 'train_batch']
    assert batch[0]['learning_rate_used'] == 0
    assert batch[1]['learning_rate_used'] == trainer.config.learning_rate
    for epoch in epochs:
        batches = [e for e in batch if e['epoch'] == epoch['history']['epoch']]
        assert epoch['history']['train_loss'] == sum(e['loss'] * e['samples'] for e in batches) / 4
    manifest = json_file(trainer.run.path / 'run_manifest.json')
    assert manifest['status'] == 'completed'
    assert manifest['code']['commit'] and manifest['command'] and manifest['cwd']
    saved = json_file(trainer.run.path / 'config.json')
    assert saved['training']['learning_rate'] == trainer.config.learning_rate
    assert saved['model']['hidden_size'] == 16
    assert json_file(trainer.run.path / 'environment.json')['packages']['torch']
    assert summarize_real_runs(tmp_path) == []
    trainer.load_best()
    assert not trainer.model.training
    r = trainer.evaluate()
    best = json_file(trainer.run.path / 'checkpoint_info.json')['best_value']
    assert metric_value(r, trainer.config.selection_metric) == best


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_epoch_resume_exact_model_optimizer_scheduler_rng_and_next_values(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    config = configuration(device=device)
    continuous = make(tmp_path, 'continuous', config)
    continuous.fit()
    expected_random = (random.random(), float(np.random.rand()), torch.rand(3))
    interrupted = make(tmp_path, 'interrupted', config)
    interrupted.fit(stop_after_epoch=1)
    assert json_file(interrupted.run.path / 'run_manifest.json')['status'] == 'interrupted'
    path = interrupted.run.path / 'checkpoints/epoch_0001.pt'
    resumed = make(tmp_path, 'resumed', config)
    resumed.fit(resume_from=path)
    assert_nested_equal(continuous.model.state_dict(), resumed.model.state_dict())
    assert_nested_equal(continuous.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_nested_equal(continuous.scheduler.state_dict(), resumed.scheduler.state_dict())
    actual_random = (random.random(), float(np.random.rand()), torch.rand(3))
    assert_nested_equal(expected_random, actual_random)
    assert resumed.state['epoch'] == continuous.state['epoch'] == 3
    assert resumed.state['global_step'] == continuous.state['global_step'] == 6
    assert resumed.state['best_epoch'] == continuous.state['best_epoch']
    assert resumed.state['best_value'] == continuous.state['best_value']
    resumed.load_best()
    metrics = resumed.evaluate()
    assert metric_value(metrics, config.selection_metric) == resumed.state['best_value']
    crows = list(csv.DictReader((continuous.run.path / 'history.csv').open(encoding='utf-8')))
    rrows = list(csv.DictReader((resumed.run.path / 'history.csv').open(encoding='utf-8')))
    for c, r in zip(crows[1:], rrows):
        assert all(c[k] == r[k] for k in c if k != 'epoch_elapsed_seconds')
    print(f'SYNTHETIC {device}: continuous vs epoch-boundary resume: model/AdamW/scheduler/RNG bitwise equal')


@pytest.mark.parametrize('mode,metric,values,expected', [
    ('max', 'entity.micro.f1', [0.5, 0.5, 0.4], 1),
    ('min', 'loss', [0.6, 0.3, 0.3], 2)])
def test_best_tie_and_selection_direction(tmp_path, mode, metric, values, expected):
    trainer = make(tmp_path, mode, configuration(selection_mode=mode, selection_metric=metric))
    original = trainer.evaluate
    oracle = iter(values)
    def selection_fixture():
        result = original()
        value = next(oracle)
        if metric == 'loss': result['loss'] = value
        else: result['entity']['micro']['f1'] = value
        return result
    trainer.evaluate = selection_fixture  # Unit fixture for selection policy, not measured F1.
    trainer.fit()
    assert trainer.state['best_epoch'] == expected


def test_early_stop_state_and_resume_stays_stopped(tmp_path):
    config = configuration(epochs=5, early_stopping_patience=1, min_delta=1.0)
    stopped = make(tmp_path, 'stop', config)
    stopped.fit()
    assert stopped.state['best_epoch'] == 1 and stopped.state['epoch'] == 2
    assert json_file(stopped.run.path / 'run_manifest.json')['status'] == 'early_stopped'
    resumed = make(tmp_path, 'stop_resume', config)
    resumed.fit(resume_from=stopped.run.path / 'checkpoints/epoch_0002.pt')
    assert resumed.state['epoch'] == 2 and resumed.state['bad_epochs'] == 1
    assert json_file(resumed.run.path / 'run_manifest.json')['status'] == 'early_stopped'


def test_rng_capture_roundtrip_all_used_generators():
    state = rng_state()
    expected = [random.random(), float(np.random.rand()), torch.rand(4)]
    if torch.cuda.is_available(): expected.append(torch.rand(4, device='cuda'))
    restore_rng(state)
    actual = [random.random(), float(np.random.rand()), torch.rand(4)]
    if torch.cuda.is_available(): actual.append(torch.rand(4, device='cuda'))
    assert_nested_equal(expected, actual)


@pytest.mark.parametrize('changes', [{'epochs': 0}, {'epochs': True}, {'batch_size': 0},
    {'learning_rate': 0}, {'learning_rate': float('nan')}, {'weight_decay': -1},
    {'warmup_steps': -1}, {'scheduler': 'bad'}, {'adam_betas': [0.9, 1]}, {'adam_betas': None},
    {'gradient_clip_norm': float('inf')}, {'shuffle': 1}, {'early_stopping_patience': 0},
    {'precision': 'float16'}, {'selection_metric': 'fake'}, {'selection_mode': 'sideways'},
    {'artifact_kind': 'fake'}, {'seed': -1}, {'device': 'bad'}])
def test_training_config_rejects_invalid(changes):
    with pytest.raises(ValueError): replace(TrainingConfig(), **changes)


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), '0.5'])
def test_bad_selection_value(value):
    with pytest.raises(ValueError): metric_value({'entity': {'micro': {'f1': value}}}, 'entity.micro.f1')


def test_missing_selection_metric():
    with pytest.raises(ValueError, match='Missing'): metric_value({}, 'entity.micro.f1')


@pytest.mark.parametrize('kind', ['real_training', 'real_validation', 'real_inference', 'unit_test'])
def test_no_real_or_unmarked_training_even_with_synthetic_provider(tmp_path, kind):
    config = configuration()
    model, train, val = tiny_components(config)
    with pytest.raises(RuntimeError, match='disabled'):
        Trainer(model, replace(config, artifact_kind=kind), train, val, tmp_path)
    assert not list(tmp_path.iterdir())


def test_provider_kind_mismatch(tmp_path):
    model, train, val = tiny_components(configuration())
    train.metadata['artifact_kind'] = 'real_training'
    with pytest.raises(ValueError, match='mismatch'):
        Trainer(model, configuration(), train, val, tmp_path)


def test_missing_provider_kind_refused(tmp_path):
    model, train, val = tiny_components(configuration())
    train.metadata.pop('artifact_kind')
    with pytest.raises(ValueError, match='mismatch'):
        Trainer(model, configuration(), train, val, tmp_path)


def test_empty_data_and_impossible_warmup(tmp_path):
    model, train, val = tiny_components(configuration())
    train.size = 0
    with pytest.raises(ValueError, match='nonempty'):
        Trainer(model, configuration(), train, val, tmp_path)
    with pytest.raises(ValueError, match='warmup'):
        make(tmp_path, 'bad_warmup', configuration(warmup_steps=6))


def test_log_path_unwritable_and_run_no_overwrite(tmp_path):
    path = tmp_path / 'file_instead_of_directory'
    path.write_text('occupied', encoding='utf-8')
    with pytest.raises(OSError): make(path, 'bad')
    make(tmp_path, 'unique')
    with pytest.raises(FileExistsError): make(tmp_path, 'unique')


@pytest.mark.parametrize('case', ['nan_loss', 'inf_loss', 'nan_gradient', 'oom', 'missing_labels', 'mask_hole', 'wrong_count', 'missing_paths'])
def test_failures_are_rejected_and_recorded(tmp_path, case):
    trainer = make(tmp_path, case)
    if case in ('missing_labels', 'mask_hole'):
        if case == 'missing_labels': trainer.train_provider.inputs.pop('character_labels')
        else: trainer.train_provider.inputs['character_mask'][0] = torch.tensor([True, False, True])
    elif case == 'wrong_count':
        trainer.train_provider.size += 1
    elif case == 'nan_gradient':
        next(trainer.model.parameters()).register_hook(lambda g: g * float('nan'))
    elif case == 'missing_paths':
        original = trainer.model.forward
        def no_paths(**kwargs):
            out = original(**kwargs)
            out['paths'] = None
            return out
        trainer.model.forward = no_paths
    else:
        def bad_forward(**kwargs):
            if case == 'oom': raise torch.cuda.OutOfMemoryError('Synthetic injected OOM for handler unit test; not actual GPU exhaustion')
            return {'loss': torch.tensor(float('nan') if case == 'nan_loss' else float('inf'), requires_grad=True)}
        trainer.model.forward = bad_forward
    with pytest.raises((ValueError, FloatingPointError, RuntimeError, IndexError)):
        trainer.fit()
    manifest = json_file(trainer.run.path / 'run_manifest.json')
    assert manifest['status'] == 'failed'
    assert manifest['exception_type']
    assert '"event": "failure"' in (trainer.run.path / 'train.log').read_text(encoding='utf-8')


def test_invalid_checkpoint_and_config_provider_mismatch(tmp_path):
    bad = tmp_path / 'bad.pt'
    bad.write_bytes(b'not checkpoint')
    trainer = make(tmp_path, 'bad_load')
    with pytest.raises(ValueError, match='Cannot read'):
        trainer.fit(resume_from=bad)
    original = make(tmp_path, 'source')
    original.fit(stop_after_epoch=1)
    cp = original.run.path / 'checkpoints/epoch_0001.pt'
    changed = make(tmp_path, 'different_config', configuration(weight_decay=0.02))
    with pytest.raises(ValueError, match='identity mismatch'):
        changed.fit(resume_from=cp)
    changed_provider = make(tmp_path, 'different_provider')
    changed_provider.val_provider.metadata['data_version'] = 'changed'
    # Identity is fixed at construction; build again after altering metadata.
    m, tr, va = tiny_components(configuration())
    va.metadata['data_version'] = 'changed'
    changed_provider = Trainer(m, configuration(), tr, va, tmp_path, run_id='provider_mismatch')
    with pytest.raises(ValueError, match='identity mismatch'):
        changed_provider.fit(resume_from=cp)
    changed_model = make(tmp_path, 'different_model')
    changed_model.identity = 'wrong_structure'
    with pytest.raises(ValueError, match='identity mismatch'):
        changed_model.fit(resume_from=cp)


def test_scheduler_constant_and_finite_clipping(tmp_path):
    trainer = make(tmp_path, 'constant', configuration(scheduler='constant', warmup_steps=0))
    trainer.fit(stop_after_epoch=1)
    assert trainer.optimizer.param_groups[0]['lr'] == trainer.config.learning_rate


def test_duplicate_validation_ids_and_prediction_lengths_rejected(tmp_path):
    trainer = make(tmp_path, 'duplicates')
    trainer.val_provider.ids[1] = trainer.val_provider.ids[0]
    with pytest.raises(ValueError): trainer.fit()


def test_no_second_fit_same_run(tmp_path):
    trainer = make(tmp_path, 'once')
    trainer.fit(stop_after_epoch=1)
    with pytest.raises(RuntimeError, match='new run'): trainer.fit()


def test_padding_does_not_affect_trainer_validation(tmp_path):
    trainer = make(tmp_path, 'padding')
    before = trainer.evaluate()
    trainer.val_provider.inputs['character_labels'][~trainer.val_provider.inputs['character_mask']] = 10
    after = trainer.evaluate()
    assert before == after


def test_all_record_types_are_distinct(tmp_path):
    for kind in ('environment_check', 'unit_test', 'synthetic_smoke_test', 'real_training', 'real_validation', 'real_inference'):
        run = Experiment(tmp_path, kind, {}, {'test_only': True}, run_id=kind)
        assert json_file(run.path / 'run_manifest.json')['artifact_kind'] == kind
    assert summarize_real_runs(tmp_path) == []  # Merely creating a manifest is not completed real training.


def test_unwritable_train_log_still_marks_run_failed(tmp_path):
    trainer = make(tmp_path, 'unwritable_log')
    (trainer.run.path / 'train.log').mkdir()  # Deterministic OS write failure, no ACL changes.
    with pytest.raises(OSError): trainer.fit()
    assert json_file(trainer.run.path / 'run_manifest.json')['status'] == 'failed'


def test_checkpoint_schema_missing_and_corrupted_state(tmp_path):
    trainer = make(tmp_path, 'schema_source')
    trainer.fit(stop_after_epoch=1)
    path = trainer.run.path / 'checkpoints/epoch_0001.pt'
    payload = torch.load(path, weights_only=True)
    for i, update in enumerate([{'schema': 'bad'}, {'training_state': {}}, {'rng': {}}, {'artifact_kind': 'real_training'}]):
        corrupted = tmp_path / f'bad_schema_{i}.pt'
        torch.save({**payload, **update}, corrupted)
        with pytest.raises(ValueError):
            read_checkpoint(corrupted, trainer.identity, trainer.config.to_dict())


def test_missing_or_invalid_validation_metric_marks_failure(tmp_path):
    for index, value in enumerate([None, float('nan')]):
        trainer = make(tmp_path, f'bad_metric_{index}')
        original = trainer.evaluate
        def invalid_value():
            result = original()
            if value is None: result.pop('entity')
            else: result['entity']['micro']['f1'] = value
            return result
        trainer.evaluate = invalid_value
        with pytest.raises(ValueError): trainer.fit()
        assert json_file(trainer.run.path / 'run_manifest.json')['status'] == 'failed'


def test_epoch_batch_order_is_seeded_and_does_not_consume_global_rng():
    _, train, _ = tiny_components(configuration())
    before = torch.get_rng_state().clone()
    first = [b['ids'] for b in train.batches(3, 2, True, 123)]
    second = [b['ids'] for b in train.batches(3, 2, True, 123)]
    assert first == second
    assert torch.equal(before, torch.get_rng_state())
