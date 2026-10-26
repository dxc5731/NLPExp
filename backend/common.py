from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.local/demo_app'
METHODS = {'rule': '规则 mc10', 'softmax': 'BERT-Softmax', 'crf': 'BERT-CRF E5'}
ENTITY_TYPES = ['Symptom', 'Drug', 'Drug_Category', 'Medical_Examination', 'Operation']

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def uid(prefix):
    return prefix + '_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def config():
    return load(RUNTIME / 'runtime.json')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def ref(path):
    p = Path(path).resolve()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}

def save(path, value):
    """Atomic writes only to new application-owned run/state directories."""
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(tmp, p)

def env():
    c = config()
    return {**os.environ, 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1',
            'GIT_OPTIONAL_LOCKS': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'TOKENIZERS_PARALLELISM': 'false', 'IMCS_DATA_DIR': c['data_dir'],
            'BERT_MODEL_DIR': c['softmax_model_dir'], 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}

def git_identity():
    def cmd(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, env=env()).decode('utf-8').strip()
    return {'revision': cmd('rev-parse', 'HEAD'), 'branch': cmd('branch', '--show-current'),
            'dirty': bool(cmd('status', '--porcelain')), 'note': 'File snapshots identify untracked application/source code separately.'}

def asset_files(method):
    c = config()
    if method == 'rule':
        return sorted((ROOT / 'baseline/dict').glob('dict_*.txt'))
    if method == 'softmax':
        return [Path(c['softmax_checkpoint']), *sorted(Path(c['softmax_model_dir']).glob('*'))]
    if method == 'crf':
        formal = load(Path(c['crf_reproduce']) / 'training_config.json')
        tok = ROOT / formal['inputs']['tokenizer']['path']
        return [Path(c['crf_checkpoint']), Path(c['crf_reproduce']) / 'training_config.json',
                *[tok / n for n in formal['inputs']['tokenizer']['files']],
                ROOT / formal['inputs']['pretrained']['path'] / 'config.json']
    raise ValueError('unknown method')

def verify_assets(method):
    c = config(); files = asset_files(method)
    if method == 'rule' and len(files) != 5:
        raise FileNotFoundError('五类冻结词表必须完整')
    identities = [ref(p) for p in files if not p.is_dir()]
    if method != 'rule' and identities[0]['sha256'] != c[method + '_checkpoint_sha256']:
        raise ValueError('正式 checkpoint SHA-256 不匹配')
    if method == 'crf':
        cfg = load(c['crf_inference_config'])
        for name, digest in cfg['snapshot_files'].items():
            if sha(Path(c['crf_reproduce']) / name) != digest:
                raise ValueError('E5冻结源码身份不匹配: ' + name)
        if sha(Path(c['crf_reproduce'])/'training_config.json') != cfg['training_config_sha256']:
            raise ValueError('E5正式配置身份不匹配')
        formal = load(Path(c['crf_reproduce'])/'training_config.json')
        for name, digest in formal['inputs']['tokenizer']['files'].items():
            if sha(ROOT/formal['inputs']['tokenizer']['path']/name) != digest:
                raise ValueError('E5 Tokenizer身份不匹配: '+name)
        if sha(ROOT/formal['inputs']['pretrained']['path']/'config.json') != formal['inputs']['pretrained']['files']['config.json']:
            raise ValueError('E5 BERT结构配置身份不匹配')
    return identities

@contextmanager
def gpu_lock(timeout=600):
    """An OS-backed cross-process lock, released automatically on process exit."""
    p = RUNTIME / 'gpu.lock'; p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a+b') as f:
        f.seek(0, 2)
        if not f.tell(): f.write(b'0'); f.flush()
        start = time.monotonic()
        while True:
            try:
                f.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, BlockingIOError):
                if time.monotonic() - start > timeout: raise TimeoutError('GPU忙，请稍后重试')
                time.sleep(0.2)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_UN)
