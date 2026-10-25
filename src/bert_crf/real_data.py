"""Read the audited public v2 snapshot without modifying its modules or splits."""
import importlib
import json
from pathlib import Path
import sys
import types

from transformers import AutoTokenizer

from .data_adapter import V2BatchProvider
from .experiment import file_hash, fingerprint


def verify_inputs(config):
    found = {}
    for kind, entry in config.value['inputs'].items():
        base = config.path(entry['path'])
        for name, expected in entry['files'].items():
            p = base if name == '.' else base / name
            if not p.is_file() or file_hash(p) != expected:
                raise ValueError(f'Input identity mismatch: {kind}/{name}')
            found[f'{kind}/{name}'] = expected
        if base.is_dir():
            actual = {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()
                      and '__pycache__' not in p.parts and p.suffix != '.pyc'}
            if actual != set(entry['files']):
                raise ValueError(f'Unmanifested input files: {kind}')
    return found


def public_dataset_module(directory):
    directory = Path(directory)
    # Isolated package name prevents collision with both our src and prior audits.
    name = '_audited_imcs_v2_' + fingerprint(str(directory.resolve()))[:12]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
    module = importlib.import_module(name + '.dataset')
    if Path(module.__file__).resolve() != (directory / 'dataset.py').resolve():
        raise ValueError('Public module provenance mismatch')
    return module


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate input JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def build_providers(config):
    identities = verify_inputs(config)
    v = config.value
    source = config.path(v['inputs']['official_train']['path'])
    if source.name != 'IMCS-V2_train.json':
        raise ValueError('Only official train source may be read')
    data = read_json(source)
    split = read_json(config.path(v['inputs']['fixed_split']['path']))
    train_ids, val_ids = split['train'], split['val']
    if any(not isinstance(ids, list) or len(set(ids)) != len(ids) for ids in (train_ids, val_ids)):
        raise ValueError('Duplicate or invalid dialogue split IDs')
    if set(train_ids) & set(val_ids) or set(train_ids) | set(val_ids) != set(data):
        raise ValueError('Fixed split overlap or source coverage mismatch')
    module = public_dataset_module(config.path(v['inputs']['interface']['path']))
    tok = AutoTokenizer.from_pretrained(config.path(v['inputs']['tokenizer']['path']),
                                        use_fast=True, local_files_only=True)
    providers = {}
    for name in ('train', 'val'):
        if len(split[name]) != v['data']['expected_dialogues'][name]:
            raise ValueError('Fixed dialogue count mismatch')
        examples = module.collect_examples(data, split[name])
        if len(examples) != v['data']['expected_sentences'][name] or sum(len(x['text']) for x in examples) != v['data']['expected_characters'][name]:
            raise ValueError('Fixed sentence/character count mismatch')
        loader = module.build_loader(examples, tok, max_len=v['resources']['max_tokens'], num_workers=0,
                                     batch_size=config.training.batch_size, shuffle=False)
        provenance = {'artifact_kind': 'real_training' if name == 'train' else 'real_validation',
            'internal_split': name, 'source': 'official_train_only', 'data_version': v['data']['version'],
            'split_version': v['data']['split_version'], 'input_identity': identities,
            'formal_config_identity': config.identity, 'bert_revision': config.model.bert_revision,
            'tokenizer_revision': config.model.bert_revision, 'num_workers': 0,
            'sample_identity': fingerprint([(x['dialogue_id'], x['sentence_id']) for x in examples])}
        providers[name] = V2BatchProvider(loader, tok, provenance=provenance)
    train_keys = set(providers['train'].adapter.records)
    val_keys = set(providers['val'].adapter.records)
    if train_keys & val_keys:
        raise ValueError('Training/validation sentence identities overlap')
    return providers, tok


def sample_batch(provider, indices):
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Nonempty unique sample indices required')
    return provider.adapter.adapt(provider.collate([provider.adapter.dataset[i] for i in indices])).trainer_batch()
