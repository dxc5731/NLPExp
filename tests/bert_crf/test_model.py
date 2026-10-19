from dataclasses import replace

import pytest
import torch
from transformers import BertConfig, BertModel

from bert_crf.config import ModelConfig
from bert_crf.model import BertEncoder, CharacterBertCRF, CharacterCRFHead, CharacterRepresentationAdapter


class SyntheticOneToOneAdapter(CharacterRepresentationAdapter):
    """TEST ONLY: caller supplies already verified 1:1 token indices; not real alignment."""
    def forward(self, token_hidden_states, *, character_mask, token_indices):
        if token_indices.shape != character_mask.shape or token_indices.dtype != torch.long:
            raise ValueError('Test token indices must be int64 [B,C]')
        return token_hidden_states.gather(1, token_indices.unsqueeze(-1).expand(-1, -1, token_hidden_states.shape[-1]))


def tiny_model():
    config = replace(ModelConfig(), hidden_size=16, device='cpu')
    bert = BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=32), add_pooling_layer=False)
    return CharacterBertCRF(config, BertEncoder(bert), SyntheticOneToOneAdapter())


def tiny_inputs():
    return dict(input_ids=torch.tensor([[2, 5, 6, 3], [2, 7, 3, 0]]),
                attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
                character_mask=torch.tensor([[True, True], [True, False]]),
                mapping={'token_indices': torch.tensor([[1, 2], [1, 0]])},
                character_labels=torch.tensor([[1, 2], [3, 0]]), decode=True)


def test_adapter_is_explicitly_abstract():
    with pytest.raises(TypeError, match='abstract'):
        CharacterRepresentationAdapter()


def test_cpu_tiny_bert_chain_and_special_token_exclusion():
    model = tiny_model()
    data = tiny_inputs()
    result = model(**data)
    assert result['emissions'].shape == (2, 2, 11)
    assert [len(p) for p in result['paths']] == [2, 1]  # No CLS, SEP or PAD tag.
    result['loss'].backward()
    assert torch.isfinite(result['loss'])
    assert torch.isfinite(model.encoder.bert.embeddings.word_embeddings.weight.grad).all()
    assert model.encoder.bert.encoder.layer[0].attention.self.query.weight.grad.abs().sum() > 0


def test_save_load_eval_exact_outputs_and_configuration(tmp_path):
    model = tiny_model().eval()
    data = tiny_inputs()
    with torch.no_grad():
        before = model(**data)
    path = tmp_path / 'checkpoint'
    model.save_pretrained(path)
    reloaded = CharacterBertCRF.from_pretrained(path, adapter=SyntheticOneToOneAdapter()).eval()
    assert model.config == reloaded.config
    with torch.no_grad():
        after = reloaded(**data)
    assert torch.equal(before['emissions'], after['emissions'])
    assert torch.equal(before['loss'], after['loss'])
    assert before['paths'] == after['paths']
    with pytest.raises(FileExistsError):
        model.save_pretrained(path)


def test_head_prediction_without_labels():
    config = replace(ModelConfig(), hidden_size=8, device='cpu')
    head = CharacterCRFHead(config).eval()
    result = head(torch.randn(1, 2, 8), torch.ones(1, 2, dtype=torch.bool), decode=True)
    assert result['loss'] is None and len(result['paths'][0]) == 2


@pytest.mark.parametrize('change', [{'labels': ['O']}, {'num_labels': 12}, {'hidden_size': 0},
                                  {'dropout': 1.0}, {'constraint_rule_version': 'other'},
                                  {'precision': 'float16'}, {'bio_constraints': 1}])
def test_bad_config(change):
    with pytest.raises(ValueError):
        replace(ModelConfig(), **change)


def test_head_rejects_hidden_shape_and_nonfinite():
    head = CharacterCRFHead(replace(ModelConfig(), hidden_size=8, device='cpu'))
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match='configured H'):
        head(torch.zeros(1, 2, 7), mask)
    with pytest.raises(ValueError, match='Nonfinite'):
        head(torch.full((1, 2, 8), float('nan')), mask)


def test_stateful_adapter_checkpoint_refused(tmp_path):
    model = tiny_model()
    model.adapter.register_buffer('state', torch.ones(1))
    with pytest.raises(ValueError, match='stateful'):
        model.save_pretrained(tmp_path / 'checkpoint')
