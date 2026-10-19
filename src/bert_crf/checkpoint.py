"""Epoch-boundary checkpoints: model, optimizer, scheduler and all used RNGs."""
import random
import math
from pathlib import Path

import numpy as np
import torch

from .experiment import file_hash

SCHEMA = 'bert_crf_epoch_checkpoint_v1'


def rng_state():
    numpy = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [numpy[0], numpy[1].tolist(), *numpy[2:]],
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device count mismatch')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['torch_cuda']])


def save_checkpoint(path, model, optimizer, scheduler, training_state, identity, config):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Checkpoint already exists; do not overwrite history')
    temporary = path.with_name(path.name + '.partial')
    payload = {'schema': SCHEMA, 'artifact_kind': config['artifact_kind'], 'identity': identity,
               'training_config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(), 'training_state': training_state.copy(),
               'rng': rng_state(), 'amp_scaler': None}
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return file_hash(path)


def read_checkpoint(path, identity, config):
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError(f'Cannot read checkpoint: {type(exc).__name__}') from exc
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA:
        raise ValueError('Invalid checkpoint schema')
    if payload.get('identity') != identity or payload.get('training_config') != config:
        raise ValueError('Checkpoint model/config/provider identity mismatch')
    required = {'model', 'optimizer', 'scheduler', 'training_state', 'rng', 'artifact_kind'}
    if not required <= payload.keys():
        raise ValueError('Incomplete checkpoint')
    state = payload['training_state']
    if payload['artifact_kind'] != config['artifact_kind'] or not isinstance(state, dict):
        raise ValueError('Invalid checkpoint kind/state')
    state_keys = {'epoch', 'global_step', 'best_value', 'best_epoch', 'bad_epochs',
                  'best_checkpoint', 'best_checkpoint_sha256'}
    if state.keys() != state_keys or any(type(state[k]) is not int or state[k] < 0 for k in ('epoch', 'global_step', 'bad_epochs')):
        raise ValueError('Invalid checkpoint training counters')
    if type(state['best_epoch']) is not int or not 1 <= state['best_epoch'] <= state['epoch']:
        raise ValueError('Invalid checkpoint best epoch')
    if type(state['best_value']) not in (int, float) or not math.isfinite(state['best_value']):
        raise ValueError('Invalid checkpoint best metric')
    if not isinstance(payload['rng'], dict) or not {'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= payload['rng'].keys():
        raise ValueError('Invalid checkpoint RNG state')
    return payload


def restore_checkpoint(path, model, optimizer, scheduler, identity, config):
    payload = read_checkpoint(path, identity, config)
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    restore_rng(payload['rng'])
    return payload['training_state'].copy()


def load_checkpoint_model(path, model, identity, config):
    """Load only model state into a caller-built model; never creates an Adapter."""
    payload = read_checkpoint(path, identity, config)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    return payload['training_state'].copy()
