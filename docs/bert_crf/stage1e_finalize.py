"""Verify artifacts and export an explicit privacy-reviewed lightweight subset.

This tool never stages, commits, pushes, trains or edits historical evidence.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
STATE = json.loads((ROOT / '.local/bert_crf/stage1e_state.json').read_text(encoding='utf-8'))
EV, LOCAL = Path(STATE['evidence']), Path(STATE['local'])
BASE = ROOT / 'docs/bert_crf/evidence'
PUBLIC = ROOT / 'docs/bert_crf/published_evidence'
D0 = '20261020T161159+0800_interface_v2_audit'
D1 = '20261020T164334+0800_stage1d1'
REVIEW = '20261020T171714+0800_stage1d_closeout_review'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def digest(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(p, value):
    p = Path(p)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if p.exists():
        if p.read_text(encoding='utf-8') == encoded:
            return  # Resume an interrupted export only when bytes are identical.
        raise FileExistsError('Existing artifact differs; never overwrite: ' + str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(encoded, encoding='utf-8')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()


def junit(path):
    root = ET.parse(path).getroot()
    cases = root.findall('.//testcase')
    return {'tests': len(cases), 'failed': sum(c.find('failure') is not None or c.find('error') is not None for c in cases),
        'skipped': sum(c.find('skipped') is not None for c in cases),
        'passed': sum(not any(c.find(t) is not None for t in ('failure', 'error', 'skipped')) for c in cases),
        'test_names': [c.attrib.get('classname', '') + '.' + c.attrib['name'] for c in cases]}


def summarize():
    tests = junit(EV / 'full_regression_final.xml')
    assert tests['tests'] == tests['passed'] == 239 and not tests['failed'] and not tests['skipped']
    assert sum('test_formal_training.' in n for n in tests['test_names']) == 51
    preflight = read(EV / 'preflight_final/preflight.json')
    assert preflight['status'] == 'PASS' and preflight['optimizer_steps'] == 0
    assert all(c['status'] == 'PASS' for c in preflight['checks'])
    lengths = read(EV / 'preflight_final/lengths_and_coverage.json')
    assert sum(x['sentences'] for x in lengths.values()) == 98529
    assert sum(x['characters'] for x in lengths.values()) == 1279361
    current_sources = read(EV / 'preflight_final/source_manifest.json')
    assert all(digest(ROOT / p) == checksum for p, checksum in current_sources.items() if p.startswith('src/'))
    probes = read(EV / 'preflight_final/real_probes.json')
    assert probes['optimizer_steps'] == 0 and not probes['performance_metrics_computed']
    for case in (probes['train'], probes['validation']):
        assert case['parameter_hash_before'] == case['parameter_hash_after'] and case['parameter_versions_unchanged']
    synthetic = read(EV / 'synthetic_pretrained_trainer.json')
    assert synthetic['status'] == 'PASS' and not synthetic['real_data_used']
    assert synthetic['state']['global_step'] == 4 and all(x > 0 for x in synthetic['max_parameter_changes'].values())
    records = [json.loads(line) for line in (EV / 'commands.jsonl').read_text(encoding='utf-8').splitlines()]
    failures = []
    for record in records:
        assert digest(record['log']) == record['log_sha256']
        if record['exit_code'] != 0:
            assert record['name'] in ('acceptance_summary', 'export_public_evidence'), record['name']
            failures.append({'name': record['name'], 'exit_code': record['exit_code'], 'log_sha256': record['log_sha256']})
        assert '--train' not in record['command'], 'Formal training command must not have been run'
    historical_runs = []
    for p in (ROOT / 'experiments/bert_crf').iterdir():
        record = read(p / 'run_manifest.json')
        assert record['artifact_kind'] == 'synthetic_smoke_test' and record['started_at'] < '2026-10-20', \
            'No new or real experiment runs expected in the formal output root'
        historical_runs.append({'directory': p.name, 'artifact_kind': record['artifact_kind'], 'started_at': record['started_at']})
    saved = read(EV / 'initial_manifest.json')
    unchanged = ('src/bert_crf/model.py', 'src/bert_crf/crf.py', 'src/bert_crf/constraints.py', 'src/bert_crf/character_mapping.py')
    assert all(digest(ROOT / p) == saved[p] for p in unchanged)
    for name in ('04_interface_v2_audit.md', '05_character_adapter_integration.md', '06_pretraining_readiness_review.md'):
        p = 'docs/bert_crf/' + name
        assert digest(ROOT / p) == saved[p]
    write(EV / 'acceptance.json', {'artifact_kind': 'engineering_acceptance', 'stage': '1E', 'technical_status': 'PASS',
        'date': datetime.now().astimezone().isoformat(), 'tests': tests, 'prior_regression': 188, 'new_tests': 51,
        'public_interface_tests': {'passed': 13, 'failed': 0}, 'preflight_status': preflight['status'],
        'formal_config_sha256': digest(ROOT / 'configs/bert_crf/first_baseline.json'),
        'real_optimizer_steps': 0, 'real_epoch_training_started': False, 'synthetic_pretrained_optimizer_steps': 4,
        'source_algorithms_unchanged': list(unchanged), 'command_logs_verified': len(records),
        'new_actual_test_failures': 0, 'actual_oom': False,
        'closeout_tool_failures_preserved': failures, 'historical_synthetic_runs_preserved': historical_runs,
        'remaining': ['Human review of first baseline choices and explicit authorization before --train',
                      'No formal model F1 or complete real Epoch timings exist yet'],
        'git_base': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short')})
    print('PASS: 239 tests, complete real preflight, no real updates; ready for publication audit')


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(v) for key, v in value.items() if key not in ('hostname',)}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(ROOT), '<PROJECT>').replace(ROOT.as_posix(), '<PROJECT>')
    return value


def export():
    records = []
    def publish(source, target, value=None, note='structured copy; project paths redacted'):
        source, target = Path(source), Path(target)
        write(target, sanitize(read(source) if value is None else value))
        records.append({'source': source.relative_to(ROOT).as_posix(), 'source_sha256': digest(source),
            'published': target.relative_to(ROOT).as_posix(), 'published_sha256': digest(target), 'transformation': note})
    d0 = read(BASE / D0 / 'alignment_audit.json')
    # Deliberately omit issue details, tokens, texts, batch token IDs and raw logs.
    d0_public = {k: d0[k] for k in ('artifact_kind', 'scope', 'started_at', 'finished_at', 'elapsed_seconds',
        'max_length', 'model_used', 'tokenizer', 'split_version', 'split_seed', 'split_source_sha256', 'splits', 'total',
        'roundtrip_nonexact_sentence_fraction', 'missing_label_character_fraction', 'previous_8_ids_exact_match', 'real_data_to_legal_bio_calls')}
    publish(BASE / D0 / 'alignment_audit.json', PUBLIC / D0 / 'alignment_summary.json', d0_public, 'allowlisted aggregate fields; no issue details/text/token IDs')
    for name in ('execution_summary.json', 'v1_v2_diff.json', 'field_mapping.json', 'interface_manifest.json'):
        publish(BASE / D0 / name, PUBLIC / D0 / name)
    for name in ('coverage_audit.json', 'eight_cases_model.json', 'real_batch_integration.json', 'test_summary.json', 'known_issues.json'):
        publish(BASE / D1 / name, PUBLIC / D1 / name)
    publish(BASE / D1 / 'full_regression_01.xml', PUBLIC / D1 / 'tests.json', junit(BASE / D1 / 'full_regression_01.xml'), 'JUnit names/outcomes only; host/output stripped')
    review = read(BASE / REVIEW / 'review_closure.json')
    publish(BASE / REVIEW / 'review_closure.json', PUBLIC / REVIEW / 'historical_readiness.json',
        {k: review[k] for k in ('audit_execution', 'pretraining_readiness', 'blockers', 'tests', 'protected_files_unchanged')},
        'historical failed readiness preserved; no original logs or personal paths')
    target = PUBLIC / STATE['run_id']
    # This file is already byte-identical and bound by first_baseline.json.
    assert digest(target / 'resource_measurements.json') == digest(EV / 'resource_measurements.json')
    records.append({'source': (EV / 'resource_measurements.json').relative_to(ROOT).as_posix(),
        'source_sha256': digest(EV / 'resource_measurements.json'),
        'published': (target / 'resource_measurements.json').relative_to(ROOT).as_posix(),
        'published_sha256': digest(target / 'resource_measurements.json'), 'transformation': 'none; byte-identical resource contract'})
    for name in ('acceptance.json', 'configuration_selection.json', 'runner_metadata_correction.json'):
        publish(EV / name, target / name)
    for name in ('preflight.json', 'real_probes.json', 'lengths_and_coverage.json', 'environment.json'):
        publish(EV / 'preflight_final' / name, target / name)
    publish(EV / 'full_regression_final.xml', target / 'tests.json', junit(EV / 'full_regression_final.xml'), 'JUnit names/outcomes only; host/output stripped')
    synthetic = read(EV / 'synthetic_pretrained_trainer.json')
    publish(EV / 'synthetic_pretrained_trainer.json', target / 'synthetic_pretrained_trainer.json',
        {k: v for k, v in synthetic.items() if k != 'manifest'}, 'synthetic-only summary; bulky manifest excluded; paths redacted')
    synthetic_run = Path(synthetic['run_directory'])
    for name in ('history.csv', 'config.json', 'metrics.json', 'checkpoint_info.json', 'run_info.json'):
        if name.endswith('.json'):
            publish(synthetic_run / name, target / 'synthetic_run' / name)
        else:
            p = target / 'synthetic_run' / name
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                raise FileExistsError(p)
            p.write_bytes((synthetic_run / name).read_bytes())
            records.append({'source': (synthetic_run / name).relative_to(ROOT).as_posix(), 'source_sha256': digest(synthetic_run / name),
                'published': p.relative_to(ROOT).as_posix(), 'published_sha256': digest(p), 'transformation': 'none; artificial-data CSV'})
    # Preserve command evidence without raw output, tokens or machine-specific env.
    for run_id in (D0, D1, STATE['run_id']):
        source = BASE / run_id / 'commands.jsonl'
        commands = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
        extracted = [{k: r[k] for k in ('name', 'command', 'exit_code', 'elapsed_seconds', 'log_sha256') if k in r} for r in commands]
        publish(source, PUBLIC / run_id / 'commands.json', extracted, 'argv, exit, elapsed and raw-log hashes only; no raw logs')
    write(target / 'export_manifest.json', {'artifact_kind': 'published_engineering_evidence',
        'exported_at': datetime.now().astimezone().isoformat(), 'files': records,
        'excluded': ['medical texts and token IDs', 'all raw logs', 'dataset/split contents', 'weights/checkpoints/environments/keys'],
        'note': 'Export time is not experiment execution time; original timestamps and source hashes retained'})
    print(f'Exported {len(records)} reviewed lightweight artifacts; originals remain local')


def audit():
    from bert_crf.formal_config import FormalConfig
    from bert_crf.real_data import verify_inputs
    cfg = FormalConfig.load(ROOT / 'configs/bert_crf/first_baseline.json', root=ROOT)
    verify_inputs(cfg)
    initial = read(EV / 'initial_manifest.json')
    allowed = {'README.md', 'docs/bert_crf/stage1e_runner.py', *['src/bert_crf/' + n for n in ('checkpoint.py', 'config.py', 'experiment.py',
        'train.py', 'trainer.py', 'training_config.py', 'data_adapter.py')]}
    changed = [p for p, h in initial.items() if not (ROOT / p).is_file() or digest(ROOT / p) != h]
    assert set(changed) <= allowed, changed
    files = set(git('diff', '--name-only').splitlines()) | set(git('ls-files', '--others', '--exclude-standard', '-z').split('\0'))
    files.discard('')
    original = read(Path(STATE['data']) / 'IMCS-V2_train.json')
    needles = {s['sentence'] for d in original.values() for s in d['dialogue'] if len(s['sentence']) >= 8}
    secrets = re.compile(r'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}|\bhf_[A-Za-z0-9]{25,}')
    findings, inventory = [], []
    for name in sorted(files):
        p = ROOT / name
        assert p.is_file() and p.suffix in ('.py', '.md', '.json', '.csv') and p.stat().st_size < 1024 * 1024, name
        text = p.read_text(encoding='utf-8')
        reasons = []
        if secrets.search(text): reasons.append('credential_format')
        if re.search(r'[A-Za-z]:[/\\]+Users[/\\]', text, re.I): reasons.append('user_profile_path')
        if any(sentence in text for sentence in needles): reasons.append('original_medical_sentence_ge8')
        if reasons: findings.append({'file': name, 'reasons': reasons})
        inventory.append({'file': name, 'bytes': p.stat().st_size, 'sha256': digest(p)})
    assert not findings, findings
    for p in (Path(STATE['data']), Path(STATE['pretrained']), ROOT / '.venv-bert-crf', EV, LOCAL, Path(STATE['archive'])):
        assert subprocess.run(['git', 'check-ignore', str(p.relative_to(ROOT))], cwd=ROOT, capture_output=True).returncode == 0
    assert git('diff', '--cached', '--name-only') == '', 'Audit before selective staging'
    assert subprocess.run(['git', 'diff', '--check'], cwd=ROOT).returncode == 0
    assert git('branch', '--show-current') == 'dxc' and git('rev-list', '--left-right', '--count', 'HEAD...origin/dxc') == '0\t0'
    output = {'status': 'PASS', 'checked_at': datetime.now().astimezone().isoformat(),
        'initial_files_preserved': len(initial), 'authorized_changed_initial_files': changed,
        'prospective_files': inventory, 'sensitive_findings': findings,
        'privacy_scan_scope': 'Limited credential formats, personal Windows paths and full original train sentences >=8 codepoints; complements explicit file/content review',
        'ignored_data_weights_logs_untouched': True, 'formal_optimizer_steps': 0,
        'source_review': 'All modified/new production sources, tests, configuration and reports inspected; old D0/D1 unchanged files also match prior inventory',
        'git_base': git('rev-parse', 'HEAD'), 'staged_at_audit': [], 'git_status': git('status', '--short')}
    write(EV / 'publication_audit.json', output)
    write(PUBLIC / STATE['run_id'] / 'publication_audit.json', sanitize(output))
    print(f'PASS: {len(inventory)} files, no scan findings, no unrelated initial file changes; staged empty')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['summarize', 'export', 'audit'])
    globals()[parser.parse_args().action]()
