"""Verify retained evidence and publish lightweight, synthetic-only stage 1C files.

No training or dataset parsing. Explicit allowlist, no checkpoint publication.
Refuses to overwrite an export. Run after the actual test commands finish.
"""
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bert_crf.experiment import file_hash, write_json
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    state = read(ROOT / '.local/bert_crf/stage1c_state.json')
    evidence, local = Path(state['evidence']), Path(state['local'])
    published = ROOT / 'docs/bert_crf/published_evidence' / state['run_id']
    if published.exists():
        raise FileExistsError('Do not overwrite historical evidence')
    suites = ET.parse(evidence / 'full_regression.xml').getroot().findall('testsuite')
    counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    assert counts == {'tests': 130, 'failures': 0, 'errors': 0, 'skipped': 0}, counts
    smoke = read(evidence / 'pretrained_trainer_final.json')
    assert smoke['status'] == 'PASS' and smoke['real_data_used'] is False
    run = Path(smoke['run_directory'])
    for line in (evidence / 'commands.jsonl').read_text(encoding='utf-8').splitlines():
        command = json.loads(line)
        assert file_hash(command['log']) == command['sha256']
        assert command['exit_code'] == 0
    original = read(ROOT / 'docs/bert_crf/evidence/20261019T162653+0800_stage1a/interface_manifest.json')
    interface = ROOT / '.local/bert_crf/20261019T162653+0800_stage1a/interface'
    assert all(file_hash(interface / x['path']) == x['sha256'] for x in original)
    old_sources = ['src/bert_crf/' + x for x in ('model.py', 'crf.py', 'constraints.py', 'config.py')]
    for path in old_sources:
        old = subprocess.check_output(['git', 'show', '2706f86:' + path], cwd=ROOT)
        assert old.replace(b'\r\n', b'\n') == (ROOT / path).read_bytes().replace(b'\r\n', b'\n')
    tested_sources = read(run / 'source_manifest.json')
    assert all(file_hash(ROOT / name) == checksum for name, checksum in tested_sources.items())
    resumes = []
    for directory in sorted((local / 'tests-regression').glob('test_epoch_resume_exact_model_*')):
        continuous = torch.load(directory / 'continuous/checkpoints/epoch_0003.pt', weights_only=True, map_location='cpu')
        resumed = torch.load(directory / 'resumed/checkpoints/epoch_0003.pt', weights_only=True, map_location='cpu')
        equality = {key: equal(continuous[key], resumed[key]) for key in ('model', 'optimizer', 'scheduler', 'rng')}
        assert all(equality.values())
        manifests = {kind: read(directory / kind / 'run_manifest.json') for kind in ('continuous', 'interrupted', 'resumed')}
        histories = {}
        logs = {}
        for kind in manifests:
            histories[kind] = list(csv.DictReader((directory / kind / 'history.csv').open(encoding='utf-8')))
            path = directory / kind / 'train.log'
            logs[kind] = {'source': str(path), 'sha256': file_hash(path), 'content': path.read_text(encoding='utf-8')}
        assert [manifests[k]['status'] for k in manifests] == ['completed', 'interrupted', 'completed']
        resumes.append({'artifact_kind': 'synthetic_smoke_test', 'directory': str(directory),
                        'device': read(directory / 'continuous/config.json')['training']['device'],
                        'checkpoint_state_bitwise_equal': equality,
                        'continuous_state': continuous['training_state'], 'resumed_state': resumed['training_state'],
                        'manifests': manifests, 'history': histories, 'original_logs': logs,
                        'checkpoint_hashes': {k: file_hash(directory / k / 'checkpoints/epoch_0003.pt') for k in ('continuous', 'resumed')}})
    assert {r['device'] for r in resumes} == {'cpu', 'cuda'}
    write_json(evidence / 'resume_verification.json', {'kind': 'synthetic_smoke_test', 'runs': resumes,
               'scope': 'Tiny random BERT only; comparisons reread actual checkpoints; next RNG-value assertions are in pytest.'})
    write_json(evidence / 'verification.json', {'verified_at': datetime.now().astimezone().isoformat(),
               'tests': counts, 'public_interface_unchanged_files': len(original),
               'unchanged_stage1b_sources': old_sources, 'tested_current_source_count': len(tested_sources),
               'real_data_used': False, 'weights_downloaded': False,
               'canonical_run': str(run), 'checkpoint_published': False})
    published.mkdir(parents=True)
    copies = []
    def copy(source, relative):
        target = published / relative
        assert source.stat().st_size < 5_000_000
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        checksum = file_hash(source)
        assert file_hash(target) == checksum
        copies.append({'source': str(source), 'published': relative, 'bytes': source.stat().st_size, 'sha256': checksum})
    files = ['commands.jsonl', 'environment.json', 'initial_code.json', 'source_manifest.json',
             'first_framework_tests.log', 'first_framework_tests.xml', 'full_regression.log', 'full_regression.xml',
             'pretrained_trainer_smoke.log', 'pretrained_trainer_smoke.json',
             'pretrained_trainer_final.log', 'pretrained_trainer_final.json', 'stage1b_regression_gpu.json',
             'resume_verification.json', 'verification.json', 'dependency_check.log']
    for name in files:
        copy(evidence / name, name)
    for name in ('config.json', 'run_manifest.json', 'environment.json', 'train.log', 'history.csv',
                 'metrics.json', 'checkpoint_info.json', 'source_manifest.json'):
        copy(run / name, 'pretrained_run/' + name)
    write_json(published / 'export_manifest.json', {'exported_at': datetime.now().astimezone().isoformat(),
               'policy': 'Explicit synthetic-only allowlist; byte-identical copies; no data/weights/checkpoints.', 'files': copies})
    print(json.dumps({'published': str(published), 'files': len(copies), 'verification': read(evidence / 'verification.json')}, indent=2))


if __name__ == '__main__':
    main()
