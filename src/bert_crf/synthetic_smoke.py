"""SYNTHETIC TEST ENTRY ONLY. No IMCS files, no real alignment implementation."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, BertConfig, BertModel

from .config import ModelConfig, configure_runtime
from .experiment import file_hash, fingerprint, write_json
from .model import BertEncoder, CharacterBertCRF, CharacterRepresentationAdapter
from .trainer import Trainer
from .training_config import TrainingConfig, require_synthetic


class SyntheticOneToOneAdapter(CharacterRepresentationAdapter):
    """Only synthetic builders below may use this gather, after proving 1:1 offsets."""
    def forward(self, token_hidden_states, *, character_mask, token_indices):
        if token_indices.dtype != torch.long or token_indices.shape != character_mask.shape:
            raise ValueError('Synthetic mapping requires Long [B,C] indices')
        return token_hidden_states.gather(1, token_indices[..., None].expand(-1, -1, token_hidden_states.shape[-1]))


class SyntheticProvider:
    """Test-only prebuilt samples; deterministic epoch permutations, no worker RNG."""
    def __init__(self, inputs, ids, *, split, source='invented_token_ids', extra=None):
        self.inputs, self.ids, self.size = inputs, ids, len(ids)
        if self.size and any(v.shape[0] != self.size for k, v in inputs.items() if isinstance(v, torch.Tensor)):
            raise ValueError('Synthetic sample count mismatch')
        def encode(value):
            return {k: encode(v) for k, v in value.items()} if isinstance(value, dict) else value.tolist()
        self.metadata = {'artifact_kind': 'synthetic_smoke_test', 'data_version': 'synthetic_char_v1',
                         'split': split, 'ids_sha256': fingerprint(ids),
                         'data_sha256': fingerprint(encode(inputs)),
                         'alignment_version': 'SYNTHETIC_ONLY_verified_one_to_one_v1',
                         'source': source, **(extra or {})}

    def batches(self, epoch, batch_size, shuffle, seed):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(seed + epoch)).tolist() if shuffle else list(range(self.size))
        def select(value, indices):
            return {k: select(v, indices) for k, v in value.items()} if isinstance(value, dict) else value[indices]
        for start in range(0, self.size, batch_size):
            indices = order[start:start + batch_size]
            yield {'ids': [self.ids[i] for i in indices], 'inputs': select(self.inputs, indices)}


def tiny_components(training):
    require_synthetic(training)
    config = replace(ModelConfig(), artifact_kind='synthetic_smoke_test', hidden_size=16,
                     seed=training.seed, device=training.device,
                     deterministic_algorithms=training.deterministic_algorithms)
    configure_runtime(config)
    bert = BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=32), add_pooling_layer=False)
    model = CharacterBertCRF(config, BertEncoder(bert), SyntheticOneToOneAdapter())
    inputs = {'input_ids': torch.tensor([[2, 5, 6, 3, 0], [2, 7, 8, 9, 3], [2, 10, 3, 0, 0], [2, 11, 12, 3, 0]]),
              'attention_mask': torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]),
              'character_mask': torch.tensor([[True, True, False], [True, True, True], [True, False, False], [True, True, False]]),
              'character_labels': torch.tensor([[1, 2, 0], [0, 3, 4], [0, 0, 0], [5, 6, 0]]),
              'mapping': {'token_indices': torch.tensor([[1, 2, 0], [1, 2, 3], [1, 0, 0], [1, 2, 0]])}}
    train = SyntheticProvider(inputs, [('synthetic_train', str(i)) for i in range(4)], split='synthetic_train')
    val = SyntheticProvider(inputs, [('synthetic_val', str(i)) for i in range(4)], split='synthetic_val')
    return model, train, val


def pretrained_components(training, model_dir):
    require_synthetic(training)
    config = ModelConfig.load(training.model_config_path)
    config = replace(config, artifact_kind='synthetic_smoke_test', seed=training.seed, device=training.device,
                     deterministic_algorithms=training.deterministic_algorithms)
    configure_runtime(config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError('Fast Tokenizer required to verify synthetic offsets')
    texts = ['天地玄黄', '宇宙', '日月', '星辰']
    encoded = tokenizer(texts, padding=True, truncation=False, return_offsets_mapping=True,
                        return_special_tokens_mask=True, return_tensors='pt')
    mask = torch.arange(4)[None, :] < torch.tensor([len(t) for t in texts])[:, None]
    indices = torch.zeros_like(mask, dtype=torch.long)
    for b, text in enumerate(texts):
        selected = [(i, tuple(offset)) for i, (offset, special) in enumerate(zip(encoded['offset_mapping'][b].tolist(), encoded['special_tokens_mask'][b].tolist())) if not special]
        if [offset for _, offset in selected] != [(i, i + 1) for i in range(len(text))]:
            raise ValueError('Synthetic test requires exactly one Token per character; no real alignment fallback')
        indices[b, :len(text)] = torch.tensor([i for i, _ in selected])
    inputs = {k: encoded[k] for k in ('input_ids', 'attention_mask', 'token_type_ids') if k in encoded}
    inputs.update(character_mask=mask, character_labels=torch.tensor([[0, 1, 2, 0], [3, 4, 0, 0], [0, 0, 0, 0], [5, 6, 0, 0]]),
                  mapping={'token_indices': indices})
    encoder = BertEncoder.from_pretrained(config, model_dir)
    if encoder.loading_info.get('missing_keys') or encoder.loading_info.get('mismatched_keys'):
        raise ValueError('Incomplete pretrained encoder')
    model = CharacterBertCRF(config, encoder, SyntheticOneToOneAdapter())
    files = {p.name: file_hash(p) for p in Path(model_dir).iterdir() if p.name in
             ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'model.safetensors')}
    extra = {'tokenizer': {'model': config.bert_model_name, 'revision': config.bert_revision,
                          'local_path': str(Path(model_dir).resolve()), 'is_fast': True, 'file_hashes': files},
             'synthetic_texts': texts, 'label_note': 'Arbitrary test BIO; not medical entity answers'}
    train = SyntheticProvider(inputs, [('synthetic_train', str(i)) for i in range(4)], split='synthetic_train', source='real_tokenizer_on_invented_text', extra=extra)
    val = SyntheticProvider(inputs, [('synthetic_val', str(i)) for i in range(4)], split='synthetic_val', source='real_tokenizer_on_invented_text', extra=extra)
    return model, train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--pretrained-dir', required=True)
    parser.add_argument('--output-root', default='experiments/bert_crf')
    parser.add_argument('--summary', required=True)
    args = parser.parse_args()
    config = TrainingConfig.load(args.config)
    require_synthetic(config)
    if Path(args.summary).exists():
        raise FileExistsError('Do not overwrite prior synthetic summary')
    started = time.perf_counter()
    model, train, val = pretrained_components(config, args.pretrained_dir)
    trainer = Trainer(model, config, train, val, args.output_root,
                      metadata={'purpose': 'stage1c_real_pretrained_trainer_integration_on_synthetic_only',
                                'config_source': str(Path(args.config).resolve()), 'config_sha256': file_hash(args.config)})
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name in
              ('encoder.bert.embeddings.word_embeddings.weight', 'head.classifier.weight', 'head.crf.transitions')}
    state = trainer.fit()
    changes = {name: float((p.detach().cpu() - before[name]).abs().max())
               for name, p in model.named_parameters() if name in before}
    if not all(value > 0 for value in changes.values()):
        raise AssertionError('BERT, Linear and CRF must each actually update')
    trainer.load_best()
    evaluation = trainer.evaluate()
    info = {'artifact_kind': 'synthetic_smoke_test', 'run_directory': str(trainer.run.path.resolve()),
            'config': config.to_dict(), 'state': state, 'max_parameter_changes': changes,
            'best_reloaded_validation': evaluation, 'elapsed_seconds_including_load': time.perf_counter() - started,
            'status': 'PASS', 'real_data_used': False}
    write_json(args.summary, info)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
