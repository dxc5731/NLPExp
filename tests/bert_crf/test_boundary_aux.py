"""Boundary variant tests; tiny synthetic updates are never formal training."""
import copy
import csv
from dataclasses import asdict, replace
import json
import random

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel

from bert_crf.boundary import balanced_bce, boundary_targets
from bert_crf.config import BoundaryAuxConfig, BoundaryModelConfig, ModelConfig, configure_runtime
from bert_crf.formal_config import FormalConfig
from bert_crf.experiment import ROOT
from bert_crf.metrics import entities
from bert_crf.model import BertEncoder, CharacterBertCRF
from bert_crf.preflight import boundary_gradient_probe, boundary_resource_contract
from bert_crf.synthetic_smoke import tiny_components, SyntheticOneToOneAdapter
from bert_crf.trainer import Trainer
from bert_crf.training_config import TrainingConfig
from test_model import tiny_inputs
from test_trainer import assert_nested_equal


def model(aux=True, device='cpu'):
    cls = BoundaryModelConfig if aux else ModelConfig
    config = cls(artifact_kind='synthetic_smoke_test', hidden_size=16, seed=42, device=device)
    configure_runtime(config)
    encoder = BertEncoder(BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32), add_pooling_layer=False))
    return CharacterBertCRF(config, encoder, SyntheticOneToOneAdapter()).to(device)


@pytest.mark.parametrize('tags,starts,ends', [
    ([0], [], []), ([0, 0, 0], [], []), ([1], [0], [0]), ([1, 2], [0], [1]),
    ([1, 2, 2], [0], [2]), ([1, 1], [0, 1], [0, 1]), ([1, 3], [0, 1], [0, 1]),
    ([0, 7, 8], [1], [2]), ([1, 2, 0, 5, 6], [0, 3], [1, 4]),
    ([9, 10, 9, 10, 0, 3], [0, 2, 5], [1, 3, 5])])
def test_targets_exact_entities(tags, starts, ends):
    gold = torch.tensor([tags], dtype=torch.long)
    s, e = boundary_targets(gold, torch.ones_like(gold, dtype=torch.bool))
    assert s.nonzero()[:, 1].tolist() == starts
    assert e.nonzero()[:, 1].tolist() == ends
    assert s.sum() == e.sum() == len(entities(tags))
    assert gold.tolist() == [tags]


def test_dynamic_padding_unicode_positions_and_illegal_bio():
    # Raw Unicode code-point positions, including whitespace/punctuation/emoji.
    text = '甲 \U0001f600，乙'
    assert len(text) == 5
    gold = torch.tensor([[1, 2, 0, 3, 4], [9, 0, 999, -100, -100]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.bool)
    s, e = boundary_targets(gold, mask)
    assert s.tolist() == [[1., 0., 0., 1., 0.], [1., 0., 0., 0., 0.]]
    assert e.tolist() == [[0., 1., 0., 0., 1.], [1., 0., 0., 0., 0.]]
    with pytest.raises(ValueError, match='Illegal BIO'):
        boundary_targets(torch.tensor([[2]]), torch.tensor([[True]]))
    with pytest.raises(ValueError, match='prefix'):
        boundary_targets(torch.tensor([[0, 0, 0]]), torch.tensor([[True, False, True]]))


@pytest.mark.parametrize('targets', [[1., 0., 0.], [0., 0., 0.], [1., 1., 1.]])
def test_balanced_loss_formula_and_padding_gradient(targets):
    logits = torch.tensor([[2., -1., .5, float('nan')]], requires_grad=True)
    gold = torch.tensor([[*targets, float('nan')]])
    mask = torch.tensor([[True, True, True, False]])
    loss = balanced_bce(logits, gold, mask)
    raw = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], gold[mask], reduction='none')
    positive = gold[mask].bool()
    parts = [raw[group].mean() for group in (positive, ~positive) if group.any()]
    assert torch.equal(loss, torch.stack(parts).mean())
    loss.backward()
    assert torch.isfinite(loss) and logits.grad[0, 3] == 0
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_main_initialization_and_rng_exact_legacy_compatibility(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    old = model(False, device)
    cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    new = model(True, device)
    assert torch.equal(cpu, torch.get_rng_state())
    assert_nested_equal(cuda, torch.cuda.get_rng_state_all())
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    assert old.boundary_head is None and len(list(new.boundary_head.parameters())) == 4
    assert 'boundary_aux' not in asdict(old.config)
    old.eval(); new.eval()
    inputs = {k: v for k, v in tiny_inputs().items()}
    from bert_crf.trainer import move
    inputs = move(inputs, torch.device(device))
    with torch.no_grad():
        a, b = old(**inputs), new(**inputs)
    assert torch.equal(a['emissions'], b['emissions']) and a['paths'] == b['paths']
    assert torch.equal(a['loss'], b['crf_nll'])


def test_joint_loss_gradients_single_encoder_and_gold_free_crf_only_decode():
    new = model().eval()
    calls = []
    handle = new.encoder.register_forward_hook(lambda *args: calls.append(1))
    data = tiny_inputs()
    out = new(**data)
    assert len(calls) == 1
    assert torch.equal(out['boundary_loss'], (out['start_loss'] + out['end_loss']) / 2)
    assert torch.equal(out['loss'], out['crf_nll'] + .2 * out['boundary_loss'])
    diagnostic = boundary_gradient_probe(new, out)
    assert all(v > 0 for v in diagnostic['weighted_aux_to_crf_norm_ratio'].values())
    crf_grad = torch.autograd.grad(out['crf_nll'], new.encoder.bert.embeddings.word_embeddings.weight, retain_graph=True)[0]
    out['loss'].backward()
    assert not torch.equal(crf_grad, new.encoder.bert.embeddings.word_embeddings.weight.grad)
    for name, parameter in new.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for module in (new.boundary_head.start, new.boundary_head.end, new.head.classifier, new.head.crf):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 0
    handle.remove()
    with torch.no_grad():
        a = new(**{**data, 'character_labels': None})
        for p in new.boundary_head.parameters():
            p.fill_(1000)
        b = new(**{**data, 'character_labels': None})
    assert a['loss'] is None and a['paths'] == b['paths'] == out['paths']
    assert torch.equal(a['emissions'], b['emissions'])


def test_aux_save_pretrained_exact_reload(tmp_path):
    m = model().eval()
    with torch.no_grad():
        before = m(**tiny_inputs())
    m.save_pretrained(tmp_path / 'aux')
    n = CharacterBertCRF.from_pretrained(tmp_path / 'aux', adapter=SyntheticOneToOneAdapter()).eval()
    with torch.no_grad():
        after = n(**tiny_inputs())
    assert n.config == m.config
    for name in ('emissions', 'loss', 'crf_nll', 'start_logits', 'end_logits'):
        assert torch.equal(before[name], after[name])
    assert before['paths'] == after['paths']
    (tmp_path / 'aux/boundary_head.pt').unlink()
    with pytest.raises(FileNotFoundError):
        CharacterBertCRF.from_pretrained(tmp_path / 'aux', adapter=SyntheticOneToOneAdapter())


def trainer(path, name, device='cpu', aux=True):
    config = replace(TrainingConfig(), seed=42, device=device, epochs=3, learning_rate=.005)
    _, train, val = tiny_components(config)
    return Trainer(model(aux, device), config, train, val, path, run_id=name, record_validation_artifacts=True)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_aux_epoch_resume_full_state_losses_optimizer_rng_and_old_rejection(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    continuous = trainer(tmp_path, 'full', device)
    optimizer_ids = {id(p) for g in continuous.optimizer.param_groups for p in g['params']}
    assert all(id(p) in optimizer_ids for p in continuous.model.boundary_head.parameters())
    continuous.fit()
    expected_rng = (random.random(), float(np.random.rand()), torch.rand(3), torch.cuda.get_rng_state_all())
    partial = trainer(tmp_path, 'partial', device)
    partial.fit(stop_after_epoch=1)
    checkpoint = partial.run.path / 'checkpoints/epoch_0001.pt'
    resumed = trainer(tmp_path, 'resumed', device)
    resumed.fit(resume_from=checkpoint)
    actual_rng = (random.random(), float(np.random.rand()), torch.rand(3), torch.cuda.get_rng_state_all())
    assert_nested_equal(expected_rng, actual_rng)
    for a, b in ((continuous.model.state_dict(), resumed.model.state_dict()),
                 (continuous.optimizer.state_dict(), resumed.optimizer.state_dict()),
                 (continuous.scheduler.state_dict(), resumed.scheduler.state_dict())):
        assert_nested_equal(a, b)
    assert continuous.state['global_step'] == resumed.state['global_step'] == 6
    resumed.load_best()
    result = resumed.evaluate()
    assert result['loss'] == result['crf_nll']
    assert result['total_loss'] > result['crf_nll']
    assert result['boundary_auxiliary']['metrics']['start']['gold'] == 3
    rows = list(csv.DictReader((continuous.run.path / 'history.csv').open(encoding='utf-8')))
    for row in rows:
        assert float(row['train_loss']) == float(row['train_crf_nll'])
        assert float(row['val_loss']) == float(row['val_crf_nll'])
        assert abs(float(row['train_total_loss']) - float(row['train_crf_nll']) - .2 * float(row['train_boundary_loss'])) < 1e-6
    old = trainer(tmp_path, 'old', device, aux=False)
    old.fit(stop_after_epoch=1)
    from bert_crf.checkpoint import read_checkpoint
    for wrong_path, target in ((old.run.path / 'checkpoints/epoch_0001.pt', resumed), (checkpoint, old)):
        with pytest.raises(ValueError, match='identity mismatch'):
            read_checkpoint(wrong_path, target.identity, target.config.to_dict())
    old.load_best()  # legacy checkpoint remains accepted by its original identity


@pytest.mark.parametrize('change', [{'weight': .1}, {'enabled': False}, {'loss': 'bce'},
    {'decode_policy': 'boundary_vote'}, {'enabled': 1}, {'weight': float('nan')}])
def test_boundary_invalid_design_refused(change):
    with pytest.raises(ValueError):
        replace(BoundaryAuxConfig(), **change)


def test_all_historical_formal_configs_and_new_strict_schema():
    for name in ('first_baseline', 'dropout_02', 'lr_1e-5'):
        c = FormalConfig.load(ROOT / f'configs/bert_crf/{name}.json', root=ROOT)
        assert asdict(c.model) == {**c.value['model'], 'labels': tuple(c.value['model']['labels'])}
    value = json.loads((ROOT / 'configs/bert_crf/lr_1e-5.json').read_text(encoding='utf-8'))
    value['model']['schema_version'] = '1.1_boundary_aux'
    value['model']['boundary_aux'] = asdict(BoundaryAuxConfig())
    c = FormalConfig(value, root=ROOT)
    assert isinstance(c.model, BoundaryModelConfig)
    value['model']['boundary_aux'].pop('weight')
    with pytest.raises(ValueError, match='configuration fields'):
        FormalConfig(value, root=ROOT)


def test_boundary_resource_contract_json_roundtrip_preserves_strict_identity():
    c = FormalConfig.load(ROOT / 'configs/bert_crf/boundary_aux_02.json', root=ROOT)
    contract = boundary_resource_contract(c)
    assert contract == json.loads(json.dumps(contract))
    assert contract['training']['adam_betas'] == [0.9, 0.999]
    assert 'measurement_sha256' not in contract['resources']
    assert len(contract['implementation_sha256']) == 4
    changed = copy.deepcopy(contract)
    changed['training']['learning_rate'] = 2e-5
    assert changed != contract  # Normalization does not remove parameter binding.
