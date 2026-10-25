"""Stage 1D-0 isolated read-only interface audit; never imports/trains a model.

UTF-8 logs stay local because the supplied demo prints medical text.
prepare creates a new run; subsequent run commands use the retained state.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.local/bert_crf/interface_v2_state.json'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def tracked_identity():
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    return {name: digest(ROOT / name) for name in files if name}


def prepare():
    if STATE.exists():
        raise FileExistsError('Existing audit state retained; do not overwrite evidence')
    archives = list(ROOT.rglob('BERT*'))
    archives = [p for p in archives if p.name.endswith('_v2.zip') and p.is_file()]
    if len(archives) != 1:
        raise FileNotFoundError(f'Expected exactly one v2 archive, found {len(archives)}')
    archive = archives[0]
    previous = json.loads((ROOT / '.local/bert_crf/stage1a_state.json').read_text(encoding='utf-8'))
    run_id = now().strftime('%Y%m%dT%H%M%S%z') + '_interface_v2_audit'
    evidence = ROOT / 'docs/bert_crf/evidence' / run_id
    local = ROOT / '.local/bert_crf' / run_id
    evidence.mkdir(parents=True, exist_ok=False)
    local.mkdir(parents=True, exist_ok=False)
    state = {'run_id': run_id, 'started_at': now().isoformat(), 'evidence': str(evidence),
             'local': str(local), 'interface': str(local / 'interface'), 'data': previous['data'],
             'tokenizer': previous['tokenizer'], 'archive': str(archive),
             'python': str(ROOT / '.venv-bert-crf/Scripts/python.exe')}
    dump(STATE, state)
    dump(evidence / 'run_context.json', state)
    manifest = []
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir():
                continue
            target = Path(state['interface']) / item.filename
            if not target.resolve().is_relative_to(Path(state['interface']).resolve()):
                raise ValueError('Unsafe ZIP path')
            raw = z.read(item)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            manifest.append({'path': item.filename, 'bytes': len(raw), 'sha256': digest(target)})
    dump(evidence / 'interface_manifest.json', manifest)
    with zipfile.ZipFile(ROOT / 'BERT数据接口_v1.zip') as z:
        old = {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith('/')}
    new = {m['path']: m['sha256'] for m in manifest}
    dump(evidence / 'v1_v2_diff.json', {'added': sorted(new.keys() - old.keys()),
         'removed': sorted(old.keys() - new.keys()), 'changed': sorted(n for n in new.keys() & old.keys() if old[n] != new[n]),
         'unchanged': sorted(n for n in new.keys() & old.keys() if old[n] == new[n])})
    split = Path(state['interface']) / 'code/artifacts/split_v1.json'
    train = Path(state['data']) / 'IMCS-V2_train.json'
    tokenizer_manifest = json.loads((ROOT / 'docs/bert_crf/evidence/20261019T162653+0800_stage1a/tokenizer_manifest.json').read_text(encoding='utf-8'))
    assert all(digest(Path(state['tokenizer']) / x['name']) == x['sha256'] for x in tokenizer_manifest['files'])
    assert digest(split) == digest(Path(previous['interface']) / 'code/artifacts/split_v1.json')
    assert digest(train) == json.loads(split.read_text(encoding='utf-8'))['source_sha256']
    dump(evidence / 'input_identity.json', {'v2_archive': str(archive), 'v2_archive_sha256': digest(archive),
         'v1_archive_sha256': digest(ROOT / 'BERT数据接口_v1.zip'),
         'data_archive_sha256': digest(ROOT / 'IMCS-V2-NER.zip'), 'train_source_sha256': digest(train),
         'split_sha256': digest(split), 'split_identical_v1': True,
         'tokenizer': tokenizer_manifest, 'data_policy': 'Reuse existing official train only; no dev/test parsed or extracted',
         'dependency_policy': 'No installs/upgrades; existing validated GPU torch retained instead of v2 old cu124 pin'})
    import platform
    import torch
    dump(evidence / 'environment.json', {'python': sys.version, 'executable': sys.executable, 'os': platform.platform(),
         'packages': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'tokenizers', 'numpy', 'pytest', 'huggingface-hub', 'seqeval', 'scikit-learn')},
         'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
         'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         'gpu_compute_capability': list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
         'gpu_compute_this_round': 'not run; data audit only'})
    dump(evidence / 'initial_git.json', {'branch': subprocess.check_output(['git', 'branch', '--show-current']).decode().strip(),
         'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
         'status': subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8'), 'tracked_hashes': tracked_identity()})
    print(json.dumps(state, indent=2))


def run(name, command, cwd=None):
    state = json.loads(STATE.read_text(encoding='utf-8'))
    evidence, local = Path(state['evidence']), Path(state['local'])
    logs = local / 'logs'
    logs.mkdir(exist_ok=True)
    log = logs / (name + '.log')
    env = os.environ.copy()
    overrides = {'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
        'IMCS_DATA_DIR': state['data'], 'BERT_MODEL_DIR': state['tokenizer'],
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'PYTHONPATH': str(Path(state['interface']) / 'code')}
    env.update(overrides)
    actual_cwd = str(Path(state['interface']) / 'code') if cwd == 'interface' else str(ROOT)
    started = now().isoformat()
    timer = time.perf_counter()
    with log.open('xb') as output:
        result = subprocess.run(command, cwd=actual_cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
    raw = log.read_text(encoding='utf-8')
    entry = {'name': name, 'artifact_kind': 'interface_audit', 'argv': command, 'cwd': actual_cwd,
             'started_at': started, 'finished_at': now().isoformat(), 'elapsed_seconds': time.perf_counter() - timer,
             'exit_code': result.returncode, 'raw_log_local_only': str(log), 'log_sha256': digest(log),
             'env_overrides': overrides, 'demo_roundtrip_true': ('一致: True' in raw) if name == 'demo' else None}
    with (evidence / 'commands.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(json.dumps(entry, indent=2, ensure_ascii=False))
    if name not in ('demo',):
        print(raw[-6000:])
    return result.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        name, *command = args.rest
        cwd = None
        if command[0] == '--interface':
            cwd = 'interface'
            command.pop(0)
        if command[0] == '--':
            command.pop(0)
        sys.exit(run(name, command, cwd))
