"""Append-only run events and unique experiment directories with source evidence."""
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

import torch

from .training_config import KINDS

ROOT = Path(__file__).resolve().parents[2]
HISTORY_FIELDS = ['epoch', 'train_loss', 'val_loss', 'val_char_precision', 'val_char_recall',
                  'val_char_f1', 'val_entity_precision', 'val_entity_recall', 'val_entity_f1',
                  'learning_rate', 'epoch_elapsed_seconds', 'global_step']
BOUNDARY_HISTORY_FIELDS = [f'{split}_{name}' for split in ('train', 'val')
    for name in ('crf_nll', 'start_loss', 'end_loss', 'boundary_loss', 'total_loss')] + ['cumulative_fit_seconds']


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def environment():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'packages': {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'tokenizers', 'numpy', 'pytest')},
            'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'capability': list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}


def code_snapshot(directory):
    directory = Path(directory)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()
    sources = sorted({*ROOT.glob('src/bert_crf/*.py'), *ROOT.glob('tests/bert_crf/*.py'),
                      *ROOT.glob('configs/bert_crf/*.json'), *ROOT.glob('docs/bert_crf/stage1c*.py'),
                      ROOT / 'pytest.ini'})
    hashes = {}
    for p in sources:
        name = p.relative_to(ROOT)
        target = directory / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        hashes[name.as_posix()] = file_hash(target)
    write_json(directory / 'source_manifest.json', hashes)
    (directory / 'tracked_changes.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD', '--', 'src/bert_crf', 'tests/bert_crf', 'configs/bert_crf'], cwd=ROOT))
    return {'commit': git('rev-parse', 'HEAD'), 'dirty_status': git('status', '--porcelain'),
            'source_manifest_sha256': file_hash(directory / 'source_manifest.json'),
            'source_snapshot': 'source_snapshot/', 'tracked_patch': 'tracked_changes.patch'}


class Experiment:
    def __init__(self, parent, artifact_kind, config, metadata, run_id=None, *, extra_history_fields=()):
        if artifact_kind not in KINDS:
            raise ValueError('Unknown run type')
        self.kind = artifact_kind
        self.history_fields = [*HISTORY_FIELDS, *extra_history_fields]
        self.log_name = 'run.log' if artifact_kind == 'real_training' else 'train.log'
        run_id = run_id or datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z') + '_' + artifact_kind + '_' + uuid.uuid4().hex[:8]
        if Path(run_id).name != run_id or run_id in ('.', '..'):
            raise ValueError('run_id must be a single directory name')
        self.path = Path(parent) / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / 'checkpoints').mkdir()
        write_json(self.path / 'config.json', config)
        write_json(self.path / 'environment.json', environment())
        self.manifest = {'schema_version': '1.0', 'artifact_kind': artifact_kind, 'run_id': run_id,
                         'started_at': datetime.now().astimezone().isoformat(), 'status': 'created',
                         'command': [sys.executable, *sys.argv], 'cwd': str(Path.cwd()),
                         'metadata': metadata, 'code': code_snapshot(self.path)}
        write_json(self.path / 'run_manifest.json', self.manifest)
        with (self.path / 'history.csv').open('x', encoding='utf-8', newline='') as stream:
            csv.DictWriter(stream, self.history_fields).writeheader()

    def status(self, status, **fields):
        self.manifest.update(status=status, updated_at=datetime.now().astimezone().isoformat(), **fields)
        write_json(self.path / 'run_manifest.json', self.manifest)

    def event(self, event, **fields):
        record = {'artifact_kind': self.kind, 'time': datetime.now().astimezone().isoformat(), 'event': event, **fields}
        with (self.path / self.log_name).open('a', encoding='utf-8') as stream:
            stream.write(canonical(record) + '\n')

    def epoch(self, row, metrics):
        self.event('epoch_complete', history=row, validation=metrics)
        with (self.path / 'history.csv').open('a', encoding='utf-8', newline='') as stream:
            csv.DictWriter(stream, self.history_fields).writerow(row)
        write_json(self.path / 'metrics.json', {'artifact_kind': self.kind, 'epoch': row['epoch'], 'validation': metrics})


def summarize_real_runs(parent):
    """Only completed real_training records are eligible for later real tables."""
    result = []
    for p in Path(parent).glob('*/run_manifest.json'):
        record = json.loads(p.read_text(encoding='utf-8'))
        if record['artifact_kind'] == 'real_training' and record['status'] in ('completed', 'early_stopped'):
            result.append(record)
    return result
