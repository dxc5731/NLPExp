"""Stage 1B evidence runner; never reads medical datasets."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.local/bert_crf/stage1b_state.json'
REVISION = '8f23c25b06e129b6c986331a13d8d025a92cf0ea'
MODEL_ID = 'google-bert/bert-base-chinese'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def state():
    return json.loads(STATE.read_text(encoding='utf-8'))


def prepare():
    if STATE.exists():
        raise FileExistsError('Use existing stage1b state; do not overwrite evidence')
    run_id = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S%z') + '_stage1b'
    evidence = ROOT / 'docs/bert_crf/evidence' / run_id
    local = ROOT / '.local/bert_crf' / run_id
    evidence.mkdir(parents=True)
    local.mkdir(parents=True)
    s = {'artifact_kind': 'stage1b_core_smoke_test', 'run_id': run_id,
         'evidence': str(evidence), 'local': str(local),
         'model': str(local / 'pretrained'), 'start_time': now()}
    write_json(STATE, s)
    import torch
    import transformers
    import tokenizers
    write_json(evidence / 'run_context.json', {
        **s, 'cwd': str(ROOT), 'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
        'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0),
        'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        'model_id': MODEL_ID, 'revision': REVISION,
    })
    print(json.dumps(s, ensure_ascii=False, indent=2))


def download():
    s = state()
    dst = Path(s['model'])
    dst.mkdir(exist_ok=True)
    old = ROOT / '.local/bert_crf/20261019T162653+0800_stage1a/tokenizer'
    files = []
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'model.safetensors', 'README.md']:
        target = dst / name
        url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
        start = time.perf_counter()
        if target.exists():
            action = 'reused_stage1b_local'
        elif (old / name).exists():
            shutil.copyfile(old / name, target)
            action = 'copied_verified_stage1a_file'
        else:
            print('Downloading', name, flush=True)
            temporary = target.with_suffix(target.suffix + '.partial')
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            temporary.replace(target)
            action = 'downloaded_pinned_public_file'
        files.append({'name': name, 'url': url, 'path': str(target), 'action': action,
                      'bytes': target.stat().st_size, 'sha256': sha(target),
                      'elapsed_seconds': time.perf_counter() - start})
        print(name, files[-1]['bytes'], files[-1]['sha256'], flush=True)
    write_json(Path(s['evidence']) / 'pretrained_manifest.json', {
        'artifact_kind': s['artifact_kind'], 'model_id': MODEL_ID, 'revision': REVISION,
        'time': now(), 'license': 'Apache-2.0 (upstream model card)', 'files': files})


def run(name, command):
    s = state()
    evidence = Path(s['evidence'])
    log = evidence / (name + '.log')
    if log.exists():
        raise FileExistsError(log)
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONPATH=str(ROOT / 'src'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    start = time.perf_counter()
    started = now()
    with log.open('wb') as out:
        p = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    result = {'name': name, 'argv': command, 'cwd': str(ROOT), 'start': started,
              'end': now(), 'elapsed_seconds': time.perf_counter() - start,
              'returncode': p.returncode, 'log': str(log), 'log_sha256': sha(log),
              'environment_overrides': {k: env[k] for k in ['PYTHONIOENCODING','PYTHONUTF8','PYTHONPATH','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM','CUBLAS_WORKSPACE_CONFIG']}}
    with (evidence / 'commands.jsonl').open('a', encoding='utf-8') as out:
        out.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(log.read_text(encoding='utf-8'))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return p.returncode


def snapshot():
    s = state()
    paths = [*ROOT.glob('src/bert_crf/*.py'), *ROOT.glob('tests/bert_crf/*.py'),
             *ROOT.glob('configs/bert_crf/*.json'), *ROOT.glob('docs/bert_crf/stage1b*.py'),
             ROOT / 'pytest.ini']
    write_json(Path(s['evidence']) / 'source_manifest.json', {
        str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)})
    print('Source hashes saved:', len(paths))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'download', 'run', 'snapshot'])
    parser.add_argument('remaining', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == 'run':
        name, *command = args.remaining
        if command and command[0] == '--':
            command.pop(0)
        sys.exit(run(name, command))
    else:
        globals()[args.action]()
