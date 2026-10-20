"""Artificial strings/labels; actual pinned Fast tokenizer, no medical data."""
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, BertConfig, BertModel

from bert_crf.character_mapping import (MAPPING_POLICY, OFFSET_UNIT, MappingError,
    OffsetCharacterAdapter, character_index)
from bert_crf.config import ModelConfig
from bert_crf.constraints import LABELS
from bert_crf.data_adapter import V2DataAdapter, V2BatchProvider, gold_ids, verify_tokenizer_coordinates
from bert_crf.model import BertEncoder, CharacterBertCRF


def mapping():
    # Token 1 covers characters 0,1. Inserted CLS/SEP and PAD have zero spans.
    return dict(offsets=torch.tensor([[[0, 0], [0, 2], [2, 3], [0, 0], [0, 0]],
                                      [[0, 0], [0, 1], [0, 0], [0, 0], [0, 0]]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
        special_tokens_mask=torch.tensor([[1, 0, 0, 1, 1], [1, 0, 1, 1, 1]], dtype=torch.bool),
        character_lengths=torch.tensor([3, 1]), offset_unit=OFFSET_UNIT, policy=MAPPING_POLICY)


def test_exact_vectors_and_shared_gradient():
    m = mapping()
    states = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4).requires_grad_()
    _, mask = character_index(**m)
    out = OffsetCharacterAdapter()(states, character_mask=mask, **m)
    assert out.shape == (2, 3, 4)
    assert torch.equal(out[0], states[0, [1, 1, 2]])
    assert torch.equal(out[1, 0], states[1, 1])
    assert torch.equal(out[1, 1:], torch.zeros(2, 4))
    out.sum().backward()
    expected = torch.zeros_like(states)
    expected[0, 1] = 2
    expected[0, 2] = 1
    expected[1, 1] = 1
    assert torch.equal(states.grad, expected)  # No CLS/SEP/PAD gradient.


@pytest.mark.parametrize('case,reason', [
    ('negative', 'invalid_offset'), ('reversed', 'invalid_offset'), ('beyond', 'invalid_offset'),
    ('gap', 'coverage_gap'), ('overlap', 'coverage_overlap'), ('order', 'offset_order'),
    ('zero_content', 'content_has_empty_span'), ('special_span', 'special_has_span'),
    ('pad_span', 'padding_has_span'), ('empty', 'empty_character_sequence'),
    ('attention_hole', 'nonprefix_attention'), ('attention_value', 'nonbinary_attention'),
    ('offset_shape', 'offset_shape_or_dtype'), ('offset_dtype', 'offset_shape_or_dtype'),
    ('attention_shape', 'attention_shape_or_dtype'), ('special_dtype', 'special_mask_shape_or_dtype'),
    ('length_shape', 'length_shape_or_dtype'), ('unit', 'unconfirmed_offset_unit_or_policy'),
    ('policy', 'unconfirmed_offset_unit_or_policy')])
def test_mapping_rejections(case, reason):
    m = mapping()
    if case == 'negative': m['offsets'][0, 1, 0] = -1
    if case == 'reversed': m['offsets'][0, 1] = torch.tensor([2, 1])
    if case == 'beyond': m['offsets'][0, 2, 1] = 4
    if case == 'gap': m['offsets'][0, 1, 1] = 1
    if case == 'overlap': m['offsets'][0, 2, 0] = 1
    if case == 'order': m['offsets'][0, 1:3] = torch.tensor([[2, 3], [0, 2]])
    if case == 'zero_content': m['offsets'][0, 1] = 0
    if case == 'special_span': m['special_tokens_mask'][0, 1] = True
    if case == 'pad_span': m['offsets'][0, 4] = torch.tensor([0, 1])
    if case == 'empty': m['character_lengths'][0] = 0
    if case == 'attention_hole': m['attention_mask'][0, 1] = 0
    if case == 'attention_value': m['attention_mask'][0, 1] = 2
    if case == 'offset_shape': m['offsets'] = m['offsets'][:, :-1, :1]
    if case == 'offset_dtype': m['offsets'] = m['offsets'].float()
    if case == 'attention_shape': m['attention_mask'] = m['attention_mask'][:, :-1]
    if case == 'special_dtype': m['special_tokens_mask'] = m['special_tokens_mask'].long()
    if case == 'length_shape': m['character_lengths'] = m['character_lengths'][:, None]
    if case == 'unit': m['offset_unit'] = 'utf16'
    if case == 'policy': m['policy'] = 'average_unknown'
    with pytest.raises(MappingError) as error:
        character_index(**m)
    assert error.value.reason == reason


@pytest.mark.parametrize('case', ['length', 'hole', 'dtype', 'shape', 'hidden_shape'])
def test_mapping_rejects_wrong_model_mask(case):
    m = mapping()
    _, mask = character_index(**m)
    states = torch.randn(2, 5, 4)
    if case == 'length': mask[0, 2] = False
    if case == 'hole': mask[0, 1] = False
    if case == 'dtype': mask = mask.long()
    if case == 'shape': mask = mask[:, :-1]
    if case == 'hidden_shape': states = states[:, :-1]
    with pytest.raises(MappingError):
        OffsetCharacterAdapter()(states, character_mask=mask, **m)


@pytest.fixture
def tokenizer():
    path = os.environ.get('BERT_MODEL_DIR') or str(Path(__file__).resolve().parents[2] / '.local/bert_crf/20261019T162653+0800_stage1a/tokenizer')
    return AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)


class ArtificialDataset:
    def __init__(self, tok, *, labeled=True):
        self.examples = [dict(dialogue_id='a', sentence_id='01', text='abc中',
                             tags=['O', 'B-Symptom', 'I-Symptom', 'O']),
                         dict(dialogue_id='b', sentence_id='01', text='药', tags=['B-Drug'])]
        self.char_labels = [ex['tags'].copy() for ex in self.examples] if labeled else None
        self.encoded = []
        for ex in self.examples:
            enc = tok(ex['text'], return_offsets_mapping=True)
            self.encoded.append({'input_ids': enc['input_ids'], 'offsets': enc['offset_mapping']})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return index


def raw_batch(ds, order=(0, 1)):
    n = max(len(ds.encoded[i]['input_ids']) for i in order)
    b = len(order)
    batch = dict(input_ids=torch.zeros(b, n, dtype=torch.long), attention_mask=torch.zeros(b, n, dtype=torch.long),
                 offsets=torch.zeros(b, n, 2, dtype=torch.long), dialogue_id=[], sentence_id=[], text=[],
                 labels=torch.full((b, n), -100, dtype=torch.long))
    for row, i in enumerate(order):
        enc, ex = ds.encoded[i], ds.examples[i]
        size = len(enc['input_ids'])
        batch['input_ids'][row, :size] = torch.tensor(enc['input_ids'])
        batch['attention_mask'][row, :size] = 1
        batch['offsets'][row, :size] = torch.tensor(enc['offsets'])
        for key in ('dialogue_id', 'sentence_id', 'text'):
            batch[key].append(ex[key])
    return batch


def test_original_gold_shuffled_composite_ids_and_token_ignore(tokenizer):
    ds = ArtificialDataset(tokenizer)
    adapter = V2DataAdapter(ds, tokenizer)
    raw = raw_batch(ds, (1, 0))
    batch = adapter.adapt(raw)
    assert batch.ids == [('b', '01'), ('a', '01')]
    assert batch.character_labels.tolist() == [[3, 0, 0, 0], [0, 1, 2, 0]]
    assert batch.character_lengths.tolist() == [1, 4]
    assert batch.character_mask.tolist() == [[True, False, False, False], [True] * 4]
    raw['labels'] = 'poison token labels are never read'
    assert torch.equal(adapter.adapt(raw).character_labels, batch.character_labels)
    assert list(batch.trainer_batch()['inputs']) == ['input_ids', 'attention_mask', 'character_mask', 'character_labels', 'mapping']


@pytest.mark.parametrize('case', ['duplicate', 'missing_gold', 'length', 'unknown_label', 'illegal_bio',
                                  'gold_mismatch', 'empty_text', 'label_order', 'id_type', 'encoded_length'])
def test_dataset_rejections(tokenizer, case):
    ds = ArtificialDataset(tokenizer)
    kwargs = {}
    if case == 'duplicate': ds.examples[1]['dialogue_id'] = 'a'
    if case == 'missing_gold': ds.char_labels[0] = None
    if case == 'length': ds.char_labels[0] = ['O']
    if case == 'unknown_label': ds.char_labels[0][0] = 'B-Unknown'
    if case == 'illegal_bio': ds.char_labels[0][0] = 'I-Symptom'
    if case == 'gold_mismatch': ds.char_labels[0][0] = 'B-Drug'
    if case == 'empty_text': ds.examples[0]['text'] = ''
    if case == 'label_order': kwargs['label_order'] = tuple(reversed(LABELS))
    if case == 'id_type': ds.examples[0]['sentence_id'] = 1
    if case == 'encoded_length': ds.encoded.pop()
    with pytest.raises(ValueError):
        V2DataAdapter(ds, tokenizer, **kwargs)


@pytest.mark.parametrize('case', ['id', 'text', 'duplicate', 'missing', 'ids', 'offset', 'attention', 'pad', 'metadata'])
def test_batch_rejections(tokenizer, case):
    ds = ArtificialDataset(tokenizer)
    adapter, batch = V2DataAdapter(ds, tokenizer), raw_batch(ds)
    if case == 'id': batch['dialogue_id'][0] = 'wrong'
    if case == 'text': batch['text'][0] = 'wrong'
    if case == 'duplicate': batch['dialogue_id'][1] = 'a'
    if case == 'missing': del batch['offsets']
    if case == 'ids': batch['input_ids'][0, 1] = 2
    if case == 'offset': batch['offsets'][0, 1, 1] += 1
    if case == 'attention': batch['attention_mask'][0, 1] = 0
    if case == 'pad': batch['input_ids'][1, -1] = 42
    if case == 'metadata': batch['text'].pop()
    with pytest.raises(ValueError):
        adapter.adapt(batch)


def test_unlabeled_and_literal_special_unicode(tokenizer):
    ds = ArtificialDataset(tokenizer, labeled=False)
    text = '[CLS]中😀，𠀀[SEP]'
    ds.examples[0]['text'] = text
    enc = tokenizer(text, return_offsets_mapping=True)
    ds.encoded[0] = {'input_ids': enc['input_ids'], 'offsets': enc['offset_mapping']}
    batch = V2DataAdapter(ds, tokenizer).adapt(raw_batch(ds))
    assert batch.character_labels is None
    assert int(batch.character_mask[0].sum()) == len(text)
    index, _ = character_index(**batch.model_inputs()['mapping'])
    assert int(index[0, 0]) == 1  # Literal [CLS] is content, even with a special ID.
    assert not bool(batch.special_tokens_mask[0, 1])
    with pytest.raises(ValueError, match='requires original'):
        batch.trainer_batch()


def test_fast_provenance_and_whitespace_failure(tokenizer):
    assert verify_tokenizer_coordinates(tokenizer)['offset_unit'] == OFFSET_UNIT
    with pytest.raises(ValueError, match='Fast'):
        verify_tokenizer_coordinates(object())
    ds = ArtificialDataset(tokenizer, labeled=False)
    ds.examples[0]['text'] = 'A B'
    enc = tokenizer('A B', return_offsets_mapping=True)
    ds.encoded[0] = {'input_ids': enc['input_ids'], 'offsets': enc['offset_mapping']}
    with pytest.raises(MappingError, match='coverage_gap'):
        V2DataAdapter(ds, tokenizer).adapt(raw_batch(ds))


def test_provider_epoch_rng_and_unchanged_real_guard(tokenizer):
    from types import SimpleNamespace
    from bert_crf.trainer import Trainer
    from bert_crf.training_config import TrainingConfig
    ds = ArtificialDataset(tokenizer)
    loader = SimpleNamespace(dataset=ds, collate_fn=lambda indices: raw_batch(ds, indices))
    provider = V2BatchProvider(loader, tokenizer, provenance={'artifact_kind': 'real_data_integration_test', 'source': 'artificial_unit_fixture'})
    rng = torch.get_rng_state().clone()
    first = list(provider.batches(1, 1, True, 8))
    assert torch.equal(rng, torch.get_rng_state())
    second = list(provider.batches(1, 1, True, 8))
    assert [b['ids'] for b in first] == [b['ids'] for b in second]
    assert {k for b in first for k in b['ids']} == {('a', '01'), ('b', '01')}
    with pytest.raises((RuntimeError, ValueError), match='synthetic|Synthetic|real'):
        Trainer(None, TrainingConfig(), provider, provider, 'must_not_create')


def test_tiny_bert_integration_and_reload(tmp_path):
    config = ModelConfig(hidden_size=16, device='cpu', dropout=0, artifact_kind='synthetic_smoke_test')
    encoder = BertEncoder(BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, hidden_dropout_prob=0, attention_probs_dropout_prob=0), add_pooling_layer=False))
    model = CharacterBertCRF(config, encoder, OffsetCharacterAdapter())
    m = mapping()
    _, mask = character_index(**m)
    inputs = dict(input_ids=torch.tensor([[2, 5, 6, 3, 0], [2, 7, 3, 0, 0]]), attention_mask=m['attention_mask'],
                  character_mask=mask, mapping=m, character_labels=torch.tensor([[0, 1, 2], [3, 0, 0]]))
    out = model(**inputs, decode=True)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    for module in (model.encoder, model.head.classifier, model.head.crf):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
        assert any(bool((g != 0).any()) for g in grads)
    assert list(map(len, out['paths'])) == [3, 1]
    for path in out['paths']:
        gold_ids([LABELS[i] for i in path], 'x' * len(path))
    model.eval()
    before = model(**inputs, decode=True)
    model.save_pretrained(tmp_path / 'checkpoint')
    loaded = CharacterBertCRF.from_pretrained(tmp_path / 'checkpoint', adapter=OffsetCharacterAdapter()).eval()
    after = loaded(**inputs, decode=True)
    assert torch.equal(before['emissions'], after['emissions'])
    assert torch.equal(before['loss'], after['loss'])
    assert before['paths'] == after['paths']
