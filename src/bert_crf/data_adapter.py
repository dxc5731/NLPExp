"""Thin adapter over A's v2 Dataset/loader, never using token-level labels.

No raw-file parsing, split generation, normalization, gold repair or training
entry point. Public modules are supplied as objects, avoiding their `src` import
namespace collision. CPU Fast Tokenizer verification establishes offset lineage.
"""
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .character_mapping import MAPPING_POLICY, OFFSET_UNIT, character_index
from .constraints import LABELS


def gold_ids(tags, text):
    if not isinstance(tags, (list, tuple)) or len(tags) != len(text):
        raise ValueError('Missing character gold or character gold length mismatch')
    ids, previous = [], 'O'
    for position, tag in enumerate(tags):
        if not isinstance(tag, str) or tag not in LABELS:
            raise ValueError(f'Unknown character gold label at position {position}')
        if tag.startswith('I-') and previous not in ('B-' + tag[2:], tag):
            raise ValueError(f'Illegal character BIO at position {position}')
        ids.append(LABELS.index(tag))
        previous = tag
    return tuple(ids)


def _key(dialogue_id, sentence_id):
    if any(not isinstance(x, str) or not x for x in (dialogue_id, sentence_id)):
        raise ValueError('Original composite IDs must be nonempty strings (no coercion)')
    return dialogue_id, sentence_id


def verify_tokenizer_coordinates(tokenizer):
    """Executable provenance check: Fast tokenizer offsets must be code points.

    Non-BMP chars distinguish Python indexing from UTF-16/UTF-8. Per-sample
    re-tokenization below verifies exact original input and inserted-special mask.
    This does not promise coverage for all Unicode inputs: gaps are rejected.
    """
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('A real Fast Tokenizer is required for offset provenance')
    text = '中😀，𠀀'
    enc = tokenizer(text, add_special_tokens=True, truncation=False,
                    return_offsets_mapping=True, return_special_tokens_mask=True)
    if len(enc['input_ids']) != len(enc['offset_mapping']) or len(enc['special_tokens_mask']) != len(enc['input_ids']):
        raise ValueError('Tokenizer metadata length mismatch')
    spans = [tuple(span) for span, special in zip(enc['offset_mapping'], enc['special_tokens_mask']) if not special]
    covered = [0] * len(text)
    for start, end in spans:
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError('Tokenizer failed Python code-point offset probe')
        for c in range(start, end):
            covered[c] += 1
    if covered != [1] * len(text) or not any(s <= 3 < e for s, e in spans):
        raise ValueError('Tokenizer failed non-BMP coordinate/coverage probe')
    return {'offset_unit': OFFSET_UNIT, 'probe': 'non_bmp_v1', 'probe_length': len(text),
            'fast_tokenizer_class': type(tokenizer).__name__}


@dataclass
class CharacterBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    offsets: torch.Tensor
    character_labels: torch.Tensor | None
    character_mask: torch.Tensor
    character_lengths: torch.Tensor
    dialogue_ids: list[str]
    sentence_ids: list[str]
    texts: list[str]
    special_tokens_mask: torch.Tensor

    @property
    def ids(self):
        return list(zip(self.dialogue_ids, self.sentence_ids))

    def model_inputs(self):
        return {'input_ids': self.input_ids, 'attention_mask': self.attention_mask,
                'character_mask': self.character_mask, 'character_labels': self.character_labels,
                'mapping': {'offsets': self.offsets, 'attention_mask': self.attention_mask,
                    'character_lengths': self.character_lengths, 'special_tokens_mask': self.special_tokens_mask,
                    'offset_unit': OFFSET_UNIT, 'policy': MAPPING_POLICY}}

    def trainer_batch(self):
        if self.character_labels is None:
            raise ValueError('A training batch requires original character gold')
        return {'ids': self.ids, 'inputs': self.model_inputs()}


class V2DataAdapter:
    def __init__(self, dataset, tokenizer, *, label_order=LABELS):
        if tuple(label_order) != LABELS:
            raise ValueError('Character label order must match the fixed CRF vocabulary')
        self.coordinate_evidence = verify_tokenizer_coordinates(tokenizer)
        self.tokenizer, self.dataset = tokenizer, dataset
        if not hasattr(dataset, 'examples') or not hasattr(dataset, 'encoded') or not hasattr(dataset, 'char_labels'):
            raise ValueError('v2 Dataset public structures missing')
        if len(dataset) != len(dataset.examples) or len(dataset.encoded) != len(dataset):
            raise ValueError('Dataset structures length mismatch')
        self.with_labels = dataset.char_labels is not None
        if hasattr(dataset, 'with_labels') and bool(dataset.with_labels) != self.with_labels:
            raise ValueError('Dataset label mode and character gold availability disagree')
        if self.with_labels and len(dataset.char_labels) != len(dataset):
            raise ValueError('Dataset character gold length mismatch')
        self.records = {}
        for i, (ex, enc) in enumerate(zip(dataset.examples, dataset.encoded)):
            key = _key(ex['dialogue_id'], ex['sentence_id'])
            if key in self.records:
                raise ValueError(f'Duplicate composite ID: {key}')
            text = ex['text']
            if not isinstance(text, str) or not text:
                raise ValueError(f'Empty/non-string original text: {key}')
            labels = None
            if self.with_labels:
                tags = dataset.char_labels[i]
                labels = gold_ids(tags, text)
                if list(tags) != ex.get('tags'):
                    raise ValueError(f'Dataset/example character gold mismatch: {key}')
            # Immutable identity snapshot, never look up by a shuffled batch index.
            self.records[key] = (text, labels, tuple(enc['input_ids']), tuple(tuple(o) for o in enc['offsets']))

    def adapt(self, batch):
        required = ('input_ids', 'attention_mask', 'offsets', 'dialogue_id', 'sentence_id', 'text')
        if any(k not in batch for k in required):
            raise ValueError('Missing v2 batch fields')
        ids, attention, offsets = (batch[k] for k in required[:3])
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 2 or 0 in ids.shape or bool((ids < 0).any()):
            raise ValueError('input_ids must be nonnegative long [B,L]')
        b, t = ids.shape
        if not isinstance(attention, torch.Tensor) or attention.shape != ids.shape or attention.dtype not in (torch.bool, torch.long):
            raise ValueError('attention shape/dtype mismatch')
        if not isinstance(offsets, torch.Tensor) or offsets.shape != (b, t, 2) or offsets.dtype != torch.long:
            raise ValueError('offset shape/dtype mismatch')
        if len({x.device for x in (ids, attention, offsets)}) != 1:
            raise ValueError('Batch tensor device mismatch')
        if any(not isinstance(batch[k], (list, tuple)) or len(batch[k]) != b for k in required[3:]):
            raise ValueError('Batch identity/text length mismatch')
        keys = [_key(d, s) for d, s in zip(batch['dialogue_id'], batch['sentence_id'])]
        if len(set(keys)) != b:
            raise ValueError('Duplicate composite ID in batch')
        special = torch.ones_like(attention, dtype=torch.bool)
        lengths, labels = [], []
        for row, key in enumerate(keys):
            if key not in self.records:
                raise ValueError(f'Unknown composite ID: {key}')
            text, gold, stored_ids, stored_offsets = self.records[key]
            if batch['text'][row] != text:
                raise ValueError(f'Batch/source original text mismatch: {key}')
            enc = self.tokenizer(text, add_special_tokens=True, truncation=False,
                return_offsets_mapping=True, return_special_tokens_mask=True)
            encoded_ids = tuple(enc['input_ids'])
            encoded_offsets = tuple(tuple(o) for o in enc['offset_mapping'])
            n = len(encoded_ids)
            if n > t or encoded_ids != stored_ids or encoded_offsets != stored_offsets:
                raise ValueError(f'Dataset/Tokenizer encoding mismatch: {key}')
            expected_attention = torch.arange(t, device=attention.device) < n
            if not torch.equal(attention[row], expected_attention.to(attention.dtype)):
                raise ValueError(f'Batch attention/Tokenizer mismatch: {key}')
            if tuple(ids[row, :n].tolist()) != encoded_ids or tuple(map(tuple, offsets[row, :n].tolist())) != encoded_offsets:
                raise ValueError(f'Batch/Tokenizer encoding mismatch: {key}')
            if bool((ids[row, n:] != self.tokenizer.pad_token_id).any()):
                raise ValueError(f'Unexpected PAD token IDs: {key}')
            special[row, :n] = torch.tensor(enc['special_tokens_mask'], dtype=torch.bool, device=ids.device)
            lengths.append(len(text))
            labels.append(gold)
        lengths = torch.tensor(lengths, dtype=torch.long, device=ids.device)
        _, mask = character_index(offsets, attention, lengths, special, offset_unit=OFFSET_UNIT)
        targets = None
        if self.with_labels:
            targets = torch.zeros_like(mask, dtype=torch.long)
            for row, gold in enumerate(labels):
                targets[row, :len(gold)] = torch.tensor(gold, dtype=torch.long, device=ids.device)
        return CharacterBatch(ids, attention, offsets, targets, mask, lengths,
            [k[0] for k in keys], [k[1] for k in keys], list(batch['text']), special)


class V2BatchProvider:
    """Reuse public Dataset/collate, with a dedicated epoch-local RNG.

    Integration probes and version-verified formal train/val are distinct modes.
    """
    def __init__(self, loader, tokenizer, *, provenance):
        kind = provenance.get('artifact_kind')
        if kind != 'real_data_integration_test':
            split = {'real_training': 'train', 'real_validation': 'val'}.get(kind)
            required = {'input_identity', 'formal_config_identity', 'sample_identity', 'bert_revision',
                        'tokenizer_revision', 'data_version', 'split_version'}
            if split is None or provenance.get('internal_split') != split or provenance.get('source') != 'official_train_only' \
                    or not required <= provenance.keys() or any(not provenance[k] for k in required):
                raise ValueError('Formal provider requires verified input/config identities; integration uses real_data_integration_test')
        self.adapter = V2DataAdapter(loader.dataset, tokenizer)
        self.size, self.collate = len(loader.dataset), loader.collate_fn
        self.metadata = {**provenance, 'mapping_policy': MAPPING_POLICY, 'offset_unit': OFFSET_UNIT,
                         'labels': list(LABELS), 'coordinate_check': self.adapter.coordinate_evidence}

    def batches(self, epoch, batch_size, shuffle, seed):
        if type(epoch) is not int or epoch < 0 or type(batch_size) is not int or batch_size <= 0:
            raise ValueError('Invalid epoch/batch_size')
        if type(seed) is not int or type(shuffle) is not bool:
            raise ValueError('Invalid seed/shuffle')
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(self.adapter.dataset, batch_size=batch_size, shuffle=shuffle,
            collate_fn=self.collate, num_workers=0, generator=generator)
        for raw in loader:
            yield self.adapter.adapt(raw).trainer_batch()
