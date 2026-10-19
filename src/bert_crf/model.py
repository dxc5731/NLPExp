"""BERT encoder, deliberately unimplemented alignment contract, character CRF head."""
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn
from transformers import BertModel

from .config import ModelConfig
from .crf import LinearChainCRF


class BertEncoder(nn.Module):
    def __init__(self, bert: BertModel):
        super().__init__()
        self.bert = bert

    @classmethod
    def from_pretrained(cls, config: ModelConfig, local_path=None):
        kwargs = {'add_pooling_layer': False}
        if local_path is None:
            kwargs['revision'] = config.bert_revision
        else:
            kwargs['local_files_only'] = True
        bert, loading_info = BertModel.from_pretrained(
            local_path or config.bert_model_name, output_loading_info=True, **kwargs)
        if bert.config.hidden_size != config.hidden_size:
            raise ValueError('BERT hidden size does not match configuration')
        encoder = cls(bert)
        encoder.loading_info = loading_info
        return encoder

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids, return_dict=True).last_hidden_state


class CharacterRepresentationAdapter(nn.Module, ABC):
    """Stage 1C contract ONLY: no real mapping implementation exists here.

    Input: token states [B,T,H], character_mask bool [B,C], and explicit
    mapping metadata from repaired shared interface. Output: [B,C,H], preserving
    every original Python Unicode code point in order (including whitespace).
    Exclude CLS/SEP/PAD. Reject missing/overlapping/out-of-range coverage or
    silent truncation; do not infer character gold from token labels.
    Mapping policy, multi-token aggregation and omitted characters await 1C.
    """
    @abstractmethod
    def forward(self, token_hidden_states, *, character_mask, **mapping):
        raise NotImplementedError('Real token-to-character alignment awaits stage 1C')


class CharacterCRFHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = LinearChainCRF(config.bio_constraints)

    def forward(self, character_hidden_states, character_mask, character_labels=None, *, decode=False):
        if character_hidden_states.ndim != 3 or character_hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError('character_hidden_states must be [B,L_char,H] with configured H')
        if not bool(torch.isfinite(character_hidden_states).all()):
            raise ValueError('Nonfinite character hidden states')
        emissions = self.classifier(self.dropout(character_hidden_states))
        self.crf._validate(emissions, character_mask, character_labels)
        return {'emissions': emissions,
                'loss': None if character_labels is None else self.crf(emissions, character_labels, character_mask),
                'paths': self.crf.decode(emissions, character_mask) if decode else None}


class CharacterBertCRF(nn.Module):
    """Orchestrates modules; caller must explicitly supply an adapter.

    Checkpoints contain encoder/head/config only. Adapter implementations are
    code dependencies, deliberately not serialized Python objects. Caller must
    supply the same adapter contract on reload; its policy belongs in run info.
    """
    def __init__(self, config, encoder, adapter):
        super().__init__()
        if encoder.bert.config.hidden_size != config.hidden_size:
            raise ValueError('Encoder/config hidden size mismatch')
        self.config = config
        self.encoder = encoder
        self.adapter = adapter
        self.head = CharacterCRFHead(config)

    def forward(self, input_ids, attention_mask, *, character_mask, mapping,
                character_labels=None, token_type_ids=None, decode=False):
        tokens = self.encoder(input_ids, attention_mask, token_type_ids)
        characters = self.adapter(tokens, character_mask=character_mask, **mapping)
        return self.head(characters, character_mask, character_labels, decode=decode)

    def save_pretrained(self, directory):
        if any(True for _ in self.adapter.parameters()) or any(True for _ in self.adapter.buffers()):
            raise ValueError('Parameterized/stateful adapters require an explicit future checkpoint schema')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.config.save(directory / 'character_crf_config.json')
        self.encoder.bert.save_pretrained(directory / 'encoder', safe_serialization=True)
        torch.save(self.head.state_dict(), directory / 'head.pt')

    @classmethod
    def from_pretrained(cls, directory, *, adapter):
        directory = Path(directory)
        config = ModelConfig.load(directory / 'character_crf_config.json')
        encoder = BertEncoder.from_pretrained(config, directory / 'encoder')
        model = cls(config, encoder, adapter)
        model.head.load_state_dict(torch.load(directory / 'head.pt', map_location='cpu', weights_only=True))
        return model
