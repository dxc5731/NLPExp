"""Integrity/traceability closure; no add, commit, push, or training."""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from stage1d1_runner import ROOT, digest, write

state = json.loads(Path(os.environ['STAGE1D1_STATE']).read_text(encoding='utf-8'))
ev = Path(state['evidence'])
protected = json.loads((ev / 'protected_manifest.json').read_text(encoding='utf-8'))
changed = [p for p, h in protected.items() if not (ROOT / p).is_file() or digest(ROOT / p) != h]
assert not changed, changed
identities = json.loads((ev / 'input_identity.json').read_text(encoding='utf-8'))
changed_inputs = [p for p, h in identities.items() if digest(p) != h]
assert not changed_inputs, changed_inputs

cases = ET.parse(ev / 'full_regression_01.xml').getroot().findall('.//testcase')
new_cases = [c for c in cases if 'test_character_adapter' in c.attrib['classname'] or 'test_character_real_integration' in c.attrib['classname']]
assert len(cases) == 188 and len(new_cases) == 58
assert all(not any(c.find(k) is not None for k in ('error', 'failure', 'skipped')) for c in cases)
write(ev / 'test_summary.json', {'total': len(cases), 'passed': len(cases), 'failed': 0, 'skipped': 0,
    'new_tests': len(new_cases), 'original_regression': len(cases) - len(new_cases),
    'first_new_tests': {'passed': 48, 'failed': 1, 'cause': 'Test assigned original offset end=3; no corruption occurred'},
    'fixed_new_tests': {'passed': 49, 'failed': 0},
    'xml_sha256': digest(ev / 'full_regression_01.xml'),
    'pretrained_trainer_evidence': 'Historical stage1c evidence revalidated; pretrained trainer optimization not rerun',
    'synthetic_regression': 'Tiny optimizer/update/resume tests executed on artificial data only; no real optimizer steps'})

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()

files = [*ROOT.glob('src/bert_crf/*adapter.py'), ROOT / 'src/bert_crf/character_mapping.py',
    *ROOT.glob('tests/bert_crf/test_character*.py'), ROOT / 'configs/bert_crf/stage1d1_integration.json',
    *ROOT.glob('docs/bert_crf/stage1d1*.py'), ROOT / 'docs/bert_crf/05_character_adapter_integration.md']
files = sorted(set(files))
manifest = {}
for p in files:
    assert p.is_file(), p
    relative = p.relative_to(ROOT)
    target = ev / 'source_snapshot' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.copyfile(p, target)
    manifest[relative.as_posix()] = digest(p)
write(ev / 'source_manifest.json', manifest)
assert len(manifest) == 9
write(ev / 'known_issues.json', {
    'resolved_test_failure': {'log': str(Path(state['local']) / 'logs/new_tests_01.log'),
        'cause': 'abc is a three-character token, so assigning endpoint 3 did not mutate offset',
        'fix': 'Increment the real endpoint instead; retain original failure and passing rerun logs',
        'gold_modified': False},
    'real_data_problems_unresolved': [],
    'limitations': ['Whitespace/normalizer gaps, overlaps, empty strings and overlong inputs reject; no fallback/windowing',
        'Characters inside a shared token have identical initial representations; quality remains unmeasured',
        'Formal Trainer remains synthetic-only; no real optimizer/resume/selection run performed',
        'No final prediction exporter or full platform dev/test inference tested',
        'Full GPU forward performed only on tiny real batches/eight anomalies, not the entire dataset'],
    'ai_use': 'Codex implemented adapters, artificial tests and audit tools; programs executed recorded checks. No research conclusion/contribution declaration written.'})
commands = [json.loads(line) for line in (ev / 'commands.jsonl').read_text(encoding='utf-8').splitlines()]
for c in commands:
    assert digest(c['log']) == c['log_sha256']
write(ev / 'final_status.json', {'time': datetime.now().astimezone().isoformat(), 'status': 'PASS',
    'branch': git('branch', '--show-current'), 'commit': git('rev-parse', 'HEAD'),
    'git_status': git('status', '--short'), 'git_mutations': [],
    'protected_file_count': len(protected), 'protected_files_changed': changed,
    'input_identities_changed': changed_inputs, 'new_stage1d1_files': manifest,
    'source_snapshot': str(ev / 'source_snapshot'), 'training_steps_on_real_data': 0,
    'command_records_before_finalization': len(commands),
    'finalization_command': 'Its exit code/log hash appended by stage1d1_runner after this script returns',
    'evidence_ignored_by_git': git('check-ignore', str(ev.relative_to(ROOT)))} )
print(json.dumps({'status': 'PASS', 'protected_files_unchanged': len(protected),
    'original_tests': len(cases) - len(new_cases), 'new_tests': len(new_cases),
    'new_source_files': len(manifest), 'branch': git('branch', '--show-current'), 'commit': git('rev-parse', 'HEAD')}, indent=2))
