"""Stage 1D-1 evidence runner. No training, Git mutations, or public-source edits."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.local/bert_crf/stage1d1_state.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def git(*args):
    p = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, encoding='utf-8', check=True)
    return p.stdout.strip()


def prepare():
    if STATE.exists():
        raise FileExistsError('Existing stage1d1 state; use its unique directories, never overwrite evidence')
    old = json.loads((ROOT / '.local/bert_crf/interface_v2_state.json').read_text(encoding='utf-8'))
    stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z') + '_stage1d1'
    state = {**old, 'run_id': stamp, 'started_at': datetime.now().astimezone().isoformat(),
             'evidence': str(ROOT / 'docs/bert_crf/evidence' / stamp),
             'local': str(ROOT / '.local/bert_crf' / stamp),
             'pretrained': str(ROOT / '.local/bert_crf/20261019T174356+0800_stage1b/pretrained'),
             'prior_issues': str(ROOT / 'docs/bert_crf/evidence/20261020T161159+0800_interface_v2_audit/alignment_issues.jsonl')}
    for key in ('evidence', 'local'):
        Path(state[key]).mkdir(parents=True, exist_ok=False)
    ev = Path(state['evidence'])
    tracked = git('ls-files', '-z').split('\0')
    protected = [p for p in tracked if p] + [
        'docs/bert_crf/04_interface_v2_audit.md', 'docs/bert_crf/interface_v2_alignment_audit.py',
        'docs/bert_crf/interface_v2_finalize.py', 'docs/bert_crf/interface_v2_runner.py']
    manifest = {p: digest(ROOT / p) for p in protected}
    interface = Path(state['interface'])
    manifest.update({str(p.relative_to(ROOT)): digest(p) for p in interface.rglob('*') if p.is_file()})
    write(ev / 'protected_manifest.json', manifest)
    write(ev / 'initial_git.json', {'branch': git('branch', '--show-current'), 'commit': git('rev-parse', 'HEAD'),
        'status': git('status', '--short'), 'protected_files': len(manifest)})
    identity_paths = [Path(state['archive']), Path(state['data']) / 'IMCS-V2_train.json',
        interface / 'code/artifacts/split_v1.json', Path(state['prior_issues'])]
    identity_paths += [p for base in ('tokenizer', 'pretrained') for p in Path(state[base]).iterdir() if p.is_file()]
    write(ev / 'input_identity.json', {str(p): digest(p) for p in identity_paths})
    write(STATE, state)
    write(ev / 'run_context.json', state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def run(name, command):
    state = json.loads(STATE.read_text(encoding='utf-8'))
    ev, local = Path(state['evidence']), Path(state['local'])
    logs = local / 'logs'
    logs.mkdir(exist_ok=True)
    log = logs / (name + '.log')
    if log.exists():
        raise FileExistsError(log)
    env = os.environ.copy()
    additions = {'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONPATH': os.pathsep.join([str(ROOT / 'src'), str(Path(state['interface']) / 'code')]),
        'IMCS_DATA_DIR': state['data'], 'BERT_MODEL_DIR': state['tokenizer'],
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'STAGE1D1_STATE': str(STATE)}
    env.update(additions)
    started = datetime.now().astimezone().isoformat()
    tick = time.perf_counter()
    with log.open('w', encoding='utf-8') as f:
        p = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    record = {'name': name, 'command': command, 'cwd': str(ROOT), 'environment_overrides': additions,
              'started_at': started, 'finished_at': datetime.now().astimezone().isoformat(),
              'elapsed_seconds': time.perf_counter() - tick, 'exit_code': p.returncode,
              'log': str(log), 'log_sha256': digest(log)}
    with (ev / 'commands.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(log.read_text(encoding='utf-8')[-16000:])
    return p.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, default=STATE,
                        help='Choose a new .local state path to create an independent repeat run')
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('name', nargs='?')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    STATE = args.state.resolve()
    if args.action == 'prepare':
        prepare()
    else:
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        sys.exit(run(args.name, command))
