"""Append-only Stage 1E command evidence; never starts formal Epoch training."""
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.local/bert_crf/stage1e_state.json'


def digest(p):
    return hashlib.file_digest(Path(p).open('rb'), 'sha256').hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()


def prepare():
    if STATE.exists():
        raise FileExistsError(STATE)
    prior = json.loads((ROOT / '.local/bert_crf/stage1d1_state.json').read_text(encoding='utf-8'))
    stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z') + '_stage1e'
    state = {**prior, 'run_id': stamp, 'started_at': datetime.now().astimezone().isoformat(), 'local': str(ROOT / '.local/bert_crf' / stamp),
             'evidence': str(ROOT / 'docs/bert_crf/evidence' / stamp)}
    for key in ('local', 'evidence'):
        Path(state[key]).mkdir(parents=True, exist_ok=False)
    files = set(git('ls-files', '-z').split('\0')) | set(git('ls-files', '--others', '--exclude-standard', '-z').split('\0'))
    manifest = {}
    for name in sorted(files - {''}):
        p = ROOT / name
        target = Path(state['local']) / 'initial_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        manifest[name] = digest(p)
    write(Path(state['evidence']) / 'initial_manifest.json', manifest)
    write(Path(state['evidence']) / 'initial_git.json', {'branch': git('branch', '--show-current'),
        'head': git('rev-parse', 'HEAD'), 'status': git('status', '--short'), 'remote': git('remote', 'get-url', 'origin')})
    write(STATE, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def run(name, command):
    state = json.loads(STATE.read_text(encoding='utf-8'))
    log = Path(state['local']) / 'logs' / (name + '.log')
    log.parent.mkdir(exist_ok=True)
    if log.exists():
        raise FileExistsError(log)
    overrides = {'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONPATH': os.pathsep.join([str(ROOT / 'src'), str(Path(state['interface']) / 'code')]),
        'IMCS_DATA_DIR': state['data'], 'BERT_MODEL_DIR': state['tokenizer'],
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'STAGE1D1_STATE': str(ROOT / '.local/bert_crf/stage1d1_state.json'), 'STAGE1E_STATE': str(STATE)}
    sources = [*ROOT.glob('src/bert_crf/*.py'), *ROOT.glob('tests/bert_crf/*.py'),
               *ROOT.glob('configs/bert_crf/*.json'), *ROOT.glob('docs/bert_crf/stage1e*.py')]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    started = datetime.now().astimezone().isoformat()
    tick = time.perf_counter()
    with log.open('x', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=ROOT, env={**os.environ, **overrides}, stdout=stream, stderr=subprocess.STDOUT)
    record = {'name': name, 'command': command, 'cwd': str(ROOT), 'environment_overrides': overrides,
        'started_at': started, 'elapsed_seconds': time.perf_counter() - tick, 'exit_code': result.returncode,
        'log': str(log), 'log_sha256': digest(log), 'source_hashes': hashes}
    with (Path(state['evidence']) / 'commands.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('source_hashes', 'environment_overrides')}, ensure_ascii=False, indent=2))
    print(log.read_text(encoding='utf-8')[-7000:])
    return result.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('name', nargs='?')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        sys.exit(run(args.name, args.command[1:] if args.command[:1] == ['--'] else args.command))
