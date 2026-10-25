"""Verify source identities and collect Stage 1B evidence (no dataset parsing)."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET

from stage1b_runner import ROOT, MODEL_ID, REVISION, now, sha, state, write_json


def collect():
    s = state()
    evidence = Path(s['evidence'])
    manifest = json.loads((evidence / 'pretrained_manifest.json').read_text(encoding='utf-8'))
    api = f'https://huggingface.co/api/models/{MODEL_ID}/tree/{REVISION}'
    tree = json.load(urllib.request.urlopen(api, timeout=60))
    upstream = next(f for f in tree if f['path'] == 'model.safetensors')
    weight = next(f for f in manifest['files'] if f['name'] == 'model.safetensors')
    assert weight['sha256'] == upstream['lfs']['oid'] == sha(weight['path'])
    assert weight['bytes'] == upstream['size']
    prior = ROOT / 'docs/bert_crf/evidence/20261019T162653+0800_stage1a'
    tokenizer = json.loads((prior / 'tokenizer_manifest.json').read_text(encoding='utf-8'))
    for f in tokenizer['files']:
        assert sha(Path(s['model']) / f['name']) == f['sha256']
    interface = ROOT / '.local/bert_crf/20261019T162653+0800_stage1a/interface'
    original_interface = json.loads((prior / 'interface_manifest.json').read_text(encoding='utf-8'))
    for f in original_interface:
        assert sha(interface / f['path']) == f['sha256']
    archives = {}
    for filename, expected in {
        'IMCS-V2-NER.zip': '84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67',
        'BERT数据接口_v1.zip': '96d259bc1b411a972b46869573f095b0af1de68e2750bdb9ba72450526723df6'
    }.items():
        archives[filename] = sha(ROOT / filename)
        assert archives[filename] == expected
    checkpoints = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                   for p in (Path(s['local']) / 'checkpoint_reporting_fix').rglob('*') if p.is_file()}
    suites = []
    for name in ['core_tests_first', 'complete_tests', 'complete_tests_reporting_fix']:
        suite = ET.parse(evidence / (name + '.xml')).getroot().find('testsuite')
        suites.append({'run': name, **suite.attrib})
    final = suites[-1]
    assert final['tests'] == '53' and final['failures'] == '0' and final['errors'] == '0' and final['skipped'] == '0'
    records = [json.loads(line) for line in (evidence / 'commands.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in records:
        assert sha(r['log']) == r['log_sha256']
    enumeration = []
    for line in (evidence / 'complete_tests_reporting_fix.log').read_text(encoding='utf-8').splitlines():
        if '{"check": "exhaustive_synthetic_paths"' in line:
            enumeration.append(json.loads(line[line.index('{'):]))
    assert len(enumeration) == 6
    write_json(evidence / 'acceptance.json', {
        'artifact_kind': s['artifact_kind'], 'time': now(), 'status': 'PASS',
        'pretrained_weight_lfs_verified': {'api': api, 'upstream': upstream, 'local': weight},
        'tokenizer_stage1a_identity': True, 'public_interface_unchanged_files': len(original_interface),
        'archive_identity': archives, 'dataset_labels_read_in_stage1b': False,
        'test_runs': suites, 'enumeration_comparisons': enumeration,
        'checkpoint_files': checkpoints, 'verified_command_logs': len(records),
        'real_character_adapter': 'NOT_IMPLEMENTED', 'optimizer_steps': 0,
        'failure_and_fix': {
            'run': 'complete_tests', 'returncode': 1,
            'cause': 'Transformers loading_info contains sets; JSON evidence serialization failed after all numerical assertions',
            'fix': 'Normalize loading_info sets to sorted lists in test evidence only',
            'before_source': 'test_pretrained_before_reporting_fix.py',
            'before_source_sha256': sha(evidence / 'test_pretrained_before_reporting_fix.py'),
            'rerun': 'complete_tests_reporting_fix', 'rerun_returncode': 0,
            'prior_checkpoint_preserved': str(Path(s['local']) / 'checkpoint')},
        'ai_use': {'tool': 'Codex', 'scope': ['core implementation', 'synthetic algorithm tests',
                   'pretrained GPU smoke', 'evidence serialization debugging', 'technical documentation'],
                   'program_verified': '53 pytest tests including real GPU and checkpoint assertions',
                   'human_review_pending': ['implementation review', 'stage1C alignment contract',
                                            'formal experimental design and research analysis'],
                   'human_modifications_this_turn': 'No independent human code modifications observed; no contribution declaration written'}
    })
    print('PASS: 53/53, pinned pretrained weight SHA-256 verified; 16 public interface files unchanged.')


def git_record():
    s = state()
    evidence = Path(s['evidence'])
    hashes = json.loads((evidence / 'source_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT / name) == value for name, value in hashes.items())
    commands = [json.loads(line) for line in (evidence / 'commands.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(sha(r['log']) == r['log_sha256'] for r in commands)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    files = subprocess.check_output(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', head], cwd=ROOT, text=True)
    write_json(Path(s['evidence']) / 'git_final.json', {
        'artifact_kind': s['artifact_kind'], 'time': now(), 'commit': head, 'status': status,
        'commit_files': files.splitlines(),
        'all_source_hashes_verified': True, 'all_completed_log_hashes_verified': True,
        'commands': [{'name': r['name'], 'returncode': r['returncode']} for r in commands],
        'source_manifest_sha256': sha(Path(s['evidence']) / 'source_manifest.json')})
    print(head, 'clean' if not status else status)


if __name__ == '__main__':
    if sys.argv[1] == 'collect':
        collect()
    elif sys.argv[1] == 'git-record':
        git_record()
    else:
        raise ValueError('Expected collect or git-record')
