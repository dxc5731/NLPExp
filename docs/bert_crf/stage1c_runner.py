"""Unique stage 1C evidence commands, UTF-8/offline; no data reads/downloads."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bert_crf.experiment import code_snapshot, environment, file_hash, write_json

STATE = ROOT / '.local/bert_crf/stage1c_state.json'


def prepare():
    if STATE.exists():
        raise FileExistsError('Do not replace existing stage1c state')
    name = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S%z') + '_stage1c'
    evidence = ROOT / 'docs/bert_crf/evidence' / name
    evidence.mkdir(parents=True)
    local = ROOT / '.local/bert_crf' / name
    local.mkdir(parents=True)
    state = {'run_id': name, 'evidence': str(evidence), 'local': str(local)}
    write_json(STATE, state)
    write_json(evidence / 'environment.json', environment())
    write_json(evidence / 'initial_code.json', code_snapshot(evidence))
    print(json.dumps(state, indent=2))


def run(name, command):
    state = json.loads(STATE.read_text(encoding='utf-8'))
    evidence = Path(state['evidence'])
    log = evidence / (name + '.log')
    if log.exists():
        raise FileExistsError(log)
    env = os.environ.copy()
    overrides = {'PYTHONPATH': str(ROOT / 'src'), 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1',
                 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    env.update(overrides)
    started = time.perf_counter()
    timestamp = datetime.now().astimezone().isoformat()
    with log.open('xb') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = {'artifact_kind': 'unit_test' if 'pytest' in command else 'environment_check',
              'name': name, 'argv': command, 'cwd': str(ROOT), 'started_at': timestamp,
              'finished_at': datetime.now().astimezone().isoformat(), 'elapsed_seconds': time.perf_counter() - started,
              'exit_code': result.returncode, 'log': str(log), 'sha256': file_hash(log), 'env': overrides}
    if 'bert_crf.synthetic_smoke' in command:
        record['artifact_kind'] = 'synthetic_smoke_test'
    with (evidence / 'commands.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(log.read_text(encoding='utf-8')[-18000:])
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return result.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        name, *command = args.rest
        if command[0] == '--': command.pop(0)
        sys.exit(run(name, command))
