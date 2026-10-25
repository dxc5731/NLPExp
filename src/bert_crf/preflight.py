"""Formal startup checks and real-data probes; never calls optimizer.step/fit."""
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .character_mapping import MAPPING_POLICY, OffsetCharacterAdapter
from .config import configure_runtime
from .experiment import code_snapshot, environment, file_hash, write_json
from .model import BertEncoder, CharacterBertCRF
from .real_data import build_providers, sample_batch
from .trainer import move


def model_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode('utf-8'))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def boundary_resource_contract(config):
    """Additional binding for the new variant; old resource identities are unchanged."""
    value = {'training': config.training.to_dict(),
        'resources': {k: v for k, v in config.value['resources'].items()
                      if k not in ('measurement_file', 'measurement_sha256')},
        'implementation_sha256': {name: file_hash(config.root / 'src/bert_crf' / name)
            for name in ('boundary.py', 'model.py', 'config.py', 'preflight.py')}}
    return json.loads(json.dumps(value, allow_nan=False))


def boundary_gradient_probe(model, output):
    """Two extra backward-only diagnostics, no update and no second BERT forward."""
    tensors = (output['character_hidden_states'], model.encoder.bert.embeddings.word_embeddings.weight)
    result = {}
    for name, loss in (('crf', output['crf_nll']),
                       ('weighted_aux', model.config.boundary_aux.weight * output['boundary_loss'])):
        gradients = torch.autograd.grad(loss, tensors, retain_graph=True)
        norms = {}
        for key, grad in zip(('shared_characters', 'bert_embeddings'), gradients):
            if not bool(torch.isfinite(grad).all()) or not bool((grad != 0).any()):
                raise FloatingPointError('Missing/nonfinite auxiliary contribution to ' + key)
            norms[key] = float(grad.detach().norm())
        result[name] = norms
        del gradients
    result['weighted_aux_to_crf_norm_ratio'] = {key: result['weighted_aux'][key] / result['crf'][key]
        for key in ('shared_characters', 'bert_embeddings')}
    return result


def build_model(config):
    configure_runtime(config.model)
    model = CharacterBertCRF(config.model,
        BertEncoder.from_pretrained(config.model, config.path(config.value['inputs']['pretrained']['path'])),
        OffsetCharacterAdapter())
    return model.to(config.training.device)


def length_summary(values, batch_size, seed, *, shuffle=True):
    a = np.asarray(values, dtype=np.int64)
    indices = torch.randperm(len(a), generator=torch.Generator().manual_seed(seed + 1)).tolist() if shuffle else list(range(len(a)))
    dynamic = sum(len(chunk) * int(a[chunk].max()) for offset in range(0, len(a), batch_size)
                  for chunk in [indices[offset:offset + batch_size]])
    return {'samples': len(values), 'min': int(a.min()), 'median': float(np.median(a)),
        **{f'p{q}': float(np.percentile(a, q, method='linear')) for q in (90, 95, 99)},
        'max': int(a.max()), 'mean': float(a.mean()),
        'histogram': {str(n): int((a == n).sum()) for n in sorted(set(values))},
        'bins': {f'{lo}-{hi}': int(((a >= lo) & (a <= hi)).sum())
                 for lo, hi in ((1, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 512))},
        'dynamic_padded_tokens': dynamic,
        'fixed_256_padded_tokens': len(values) * 256,
        'dynamic_to_fixed256_ratio': dynamic / (len(values) * 256),
        'padding_over_actual_ratio': dynamic / int(a.sum()),
        'batch_size': batch_size, 'shuffle_seed': seed + 1 if shuffle else None,
        'order': 'epoch1_seeded_training' if shuffle else 'validation_source_order'}


def scan_providers(config, providers):
    result = {}
    for name, provider in providers.items():
        lengths = [len(x['input_ids']) for x in provider.adapter.dataset.encoded]
        max_len = max(lengths)
        if max_len > config.value['resources']['max_tokens']:
            raise ValueError('Overlong sentence: no truncation allowed')
        seen, chars = set(), 0
        # Adapter verifies IDs/text/raw gold/re-tokenization/unique offsets for ALL.
        for batch in provider.batches(0, 64, False, config.training.seed):
            if seen.intersection(batch['ids']):
                raise ValueError('Repeated source IDs')
            seen.update(batch['ids'])
            chars += int(batch['inputs']['character_mask'].sum())
        if seen != set(provider.adapter.records) or chars != config.value['data']['expected_characters'][name]:
            raise ValueError('Full provider coverage/count mismatch')
        result[name] = {'sentences': len(seen), 'characters': chars, 'unique_coverage': True,
            'gold_source': 'Dataset.char_labels, composite ID verified; token labels unused',
            'token_lengths': length_summary(lengths, config.training.batch_size, config.training.seed, shuffle=name == 'train'),
            'overlong_sentences': 0, 'rejected_sentences': 0}
    return result


def probe_batch(model, provider, indices, *, backward, reserve_multiplier=0):
    """One disposable graph, with an optional real allocation for future AdamW.

    reserve=3 FP32 parameter copies covers two moments and a conservative working
    buffer. It is a capacity check, not a claim AdamW step latency was measured.
    """
    device = next(model.parameters()).device
    model.zero_grad(set_to_none=True)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    before = model_hash(model)
    versions = [p._version for p in model.parameters()]
    batch = sample_batch(provider, indices)
    inputs = move(batch['inputs'], device)
    reserve = None
    tick = time.perf_counter()
    try:
        if reserve_multiplier:
            reserve = torch.empty(sum(p.numel() for p in model.parameters()) * reserve_multiplier,
                                  dtype=torch.float32, device=device)
        model.train(backward)
        with torch.set_grad_enabled(backward):
            output = model(**inputs, decode=not backward)
            loss = output['loss']
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Probe Loss NaN/Inf')
            if backward:
                contribution = boundary_gradient_probe(model, output) if model.boundary_head is not None else None
                loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        gradients = {}
        if backward:
            modules = [('bert', model.encoder), ('linear', model.head.classifier), ('crf', model.head.crf)]
            if model.boundary_head is not None:
                modules.extend([('start_head', model.boundary_head.start), ('end_head', model.boundary_head.end)])
            for name, module in modules:
                params = [p for p in module.parameters() if p.requires_grad]
                finite = sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
                nonzero = sum(p.grad is not None and bool((p.grad != 0).any()) for p in params)
                if finite != len(params) or not nonzero:
                    raise FloatingPointError(f'Missing/nonfinite gradients: {name}')
                gradients[name] = {'tensors': len(params), 'finite': finite, 'nonzero': nonzero}
        else:
            from .metrics import checked_tags
            for row, path in enumerate(output['paths']):
                if len(path) != int(inputs['character_mask'][row].sum()):
                    raise ValueError('Probe prediction length mismatch')
                checked_tags(path)
        elapsed = time.perf_counter() - tick
        result = {'status': 'PASS', 'batch_size': len(indices),
            'token_shape': list(inputs['input_ids'].shape), 'character_shape': list(inputs['character_mask'].shape),
            'token_lengths': inputs['attention_mask'].sum(1).tolist(),
            'character_lengths': inputs['character_mask'].sum(1).tolist(),
            'loss': float(loss.detach()), 'elapsed_seconds': elapsed, 'backward': backward,
            'gradients': gradients, 'optimizer_steps': 0, 'reserve_multiplier': reserve_multiplier,
            'reserve_bytes': 0 if reserve is None else reserve.numel() * reserve.element_size(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
            'free_bytes_at_peak_end': torch.cuda.mem_get_info(device)[0] if device.type == 'cuda' else None,
            'parameter_hash_before': before, 'parameter_hash_after': model_hash(model),
            'parameter_versions_unchanged': versions == [p._version for p in model.parameters()],
            'timing_scope': 'allocation, GPU forward/backward or decode, gradient checks; excludes tokenizer, hashing, optimizer step and I/O'}
        if model.boundary_head is not None:
            from .boundary import boundary_counts
            result['loss_components'] = {key: float(output[key].detach()) for key in
                ('crf_nll', 'start_loss', 'end_loss', 'boundary_loss', 'total_loss')}
            result['weighted_boundary_loss'] = model.config.boundary_aux.weight * result['loss_components']['boundary_loss']
            if not all(math.isfinite(v) for v in result['loss_components'].values()):
                raise FloatingPointError('Nonfinite boundary loss component')
            result['boundary_counts'] = boundary_counts(output, inputs['character_labels'], inputs['character_mask'])
            result['gradient_contribution'] = contribution if backward else None
            result['decode_policy'] = 'crf_only'
        if result['parameter_hash_before'] != result['parameter_hash_after'] or not result['parameter_versions_unchanged']:
            raise AssertionError('Probe changed model parameters')
        return result
    finally:
        model.zero_grad(set_to_none=True)
        del reserve


@dataclass
class ReadyContext:
    config: object
    providers: dict
    model: object
    report: dict

    def validate_training(self, model, training, train, val):
        if self.report.get('status') != 'PASS' or self.report.get('optimizer_steps') != 0:
            raise RuntimeError('Formal training disabled: passing preflight required')
        if model is not self.model or train is not self.providers['train'] or val is not self.providers['val']:
            raise ValueError('Formal context/model/provider identity mismatch')
        if training.to_dict() != self.config.training.to_dict():
            raise ValueError('Formal context training config mismatch')
        if self.config.identity != self.report['formal_config_identity']:
            raise ValueError('Formal configuration mutated after preflight')
        for split, kind in (('train', 'real_training'), ('val', 'real_validation')):
            meta = self.providers[split].metadata
            if meta.get('artifact_kind') != kind or meta.get('formal_config_identity') != self.config.identity:
                raise ValueError('Formal provider mode/identity mismatch')
        if model.config != self.config.model:
            raise ValueError('Formal model configuration mismatch')


def check_output_paths(config):
    root = config.path(config.value['outputs']['root'])
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Unsafe experiment root')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='.preflight-', dir=root) as temp:
        p = Path(temp) / 'write_probe'
        p.write_bytes(b'preflight')
        if p.read_bytes() != b'preflight':
            raise OSError('Output write/read failed')
    if shutil.disk_usage(root).free < config.value['resources']['minimum_disk_free_bytes']:
        raise OSError('Insufficient disk for checkpoints and artifacts')


def run_preflight(config, directory):
    directory = Path(directory).resolve()
    if directory == config.root or not directory.is_relative_to(config.root):
        raise ValueError('Preflight output must be a new directory inside the project')
    protected = [config.path(config.value['outputs']['root']),
                 *[config.root / name for name in ('src', 'tests', 'configs')]]
    for entry in config.value['inputs'].values():
        p = config.path(entry['path'])
        protected.append(p.parent if '.' in entry['files'] else p)
    if any(directory == p or directory.is_relative_to(p) for p in protected):
        raise ValueError('Preflight output overlaps protected inputs/source/formal experiments')
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {'artifact_kind': 'environment_check', 'status': 'RUNNING', 'checks': [], 'optimizer_steps': 0,
        'formal_config_identity': config.identity, 'run_id': directory.name,
        'started_at': datetime.now().astimezone().isoformat()}
    def passed(name, detail):
        report['checks'].append({'check': name, 'status': 'PASS', 'detail': detail})
        write_json(directory / 'preflight.json', report)
        print('PASS ' + name, flush=True)
    try:
        configure_runtime(config.model)
        write_json(directory / 'config.json', config.value)
        write_json(directory / 'environment.json', environment())
        report['code'] = code_snapshot(directory)
        passed('configuration', 'strict explicit schema; FP32; fixed BIO and selection policy')
        check_output_paths(config)
        passed('output_paths', 'new unique run directories; writable log/checkpoint/results root; disk budget')
        providers, _ = build_providers(config)
        passed('inputs_and_providers', 'all input file hashes verified; fixed complete disjoint IDs; original character gold')
        stats = scan_providers(config, providers)
        write_json(directory / 'lengths_and_coverage.json', stats)
        passed('full_character_coverage', {k: {'sentences': x['sentences'], 'characters': x['characters']}
                                         for k, x in stats.items()})
        r = config.value['resources']
        measurement = config.path(r['measurement_file'])
        if file_hash(measurement) != r['measurement_sha256']:
            raise ValueError('Resource measurement hash mismatch')
        import json
        measured = json.loads(measurement.read_text(encoding='utf-8'))
        from dataclasses import asdict
        if measured['model_config'] != json.loads(json.dumps(asdict(config.model))) or measured['inputs'] != config.value['inputs']:
            raise ValueError('Resource evidence model/input configuration mismatch')
        if hasattr(config.model, 'boundary_aux'):
            if measured.get('boundary_resource_contract') != boundary_resource_contract(config):
                raise ValueError('Boundary resource evidence implementation/training contract mismatch')
        now = environment()
        if any(measured['environment'][key] != now[key] for key in ('packages', 'gpu', 'cuda_runtime', 'capability')):
            raise ValueError('Resource evidence hardware/software mismatch; remeasure')
        selected = [case for case in measured['cases'] if case['batch_size'] == config.training.batch_size
                    and case['sample_strategy'] == 'longest_unique' and case['status'] == 'PASS'
                    and case['reserve_multiplier'] == 3]
        if len(selected) != 1 or selected[0]['max_tokens'] != r['max_tokens']:
            raise ValueError('Selected batch/resource strategy not measured')
        selected = selected[0]
        if hasattr(config.model, 'boundary_aux'):
            if not selected.get('gradient_contribution') or not selected.get('loss_components') or \
                    any(selected['gradients'].get(name, {}).get('nonzero', 0) == 0 for name in ('start_head', 'end_head')):
                raise ValueError('Boundary resource evidence lacks real auxiliary forward/backward')
        if measured.get('optimizer_steps') != 0 or selected.get('optimizer_steps') != 0 \
                or selected['parameter_hash_before'] != selected['parameter_hash_after'] \
                or not selected['parameter_versions_unchanged'] \
                or selected['token_shape'][1] < stats['train']['token_lengths']['max']:
            raise ValueError('Resource evidence lacks zero-update/full-length verification')
        free, total = torch.cuda.mem_get_info() if config.training.device == 'cuda' else (0, 0)
        if config.training.device != 'cuda' or free < selected['peak_reserved_bytes'] + r['minimum_free_bytes']:
            raise RuntimeError('GPU free memory insufficient for measured peak plus required margin')
        passed('resource_contract', {'batch_size': config.training.batch_size, 'gpu_total_bytes': total, 'free_bytes': free})
        model = build_model(config)
        order = sorted(range(providers['train'].size), key=lambda i: len(providers['train'].adapter.dataset.encoded[i]['input_ids']), reverse=True)
        train_probe = probe_batch(model, providers['train'], order[:config.training.batch_size], backward=True, reserve_multiplier=3)
        if train_probe['free_bytes_at_peak_end'] < r['minimum_free_bytes']:
            raise RuntimeError('GPU reserve/margin check failed during actual longest-batch probe')
        val_probe = probe_batch(model, providers['val'], list(range(config.training.batch_size)), backward=False)
        write_json(directory / 'real_probes.json', {'artifact_kind': 'real_data_integration_test',
            'train': train_probe, 'validation': val_probe, 'optimizer_steps': 0, 'performance_metrics_computed': False})
        passed('real_forward_backward_and_validation', 'finite loss/gradients, legal equal-length decode, parameter hashes unchanged')
        configure_runtime(config.model)  # probes must not consume training's initial RNG stream
        model.train()
        report.update(status='PASS', completed_at=datetime.now().astimezone().isoformat(),
                      elapsed_seconds=time.perf_counter() - started)
        write_json(directory / 'preflight.json', report)
        return ReadyContext(config, providers, model, report)
    except BaseException as exc:
        report.update(status='FAIL', exception_type=type(exc).__name__, error=str(exc),
                      elapsed_seconds=time.perf_counter() - started)
        report['checks'].append({'check': 'startup', 'status': 'FAIL', 'detail': str(exc)})
        write_json(directory / 'preflight.json', report)
        print('FAIL startup: ' + type(exc).__name__ + ': ' + str(exc), flush=True)
        raise
