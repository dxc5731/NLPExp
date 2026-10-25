"""Readiness review evidence, including intentional rejection probes.

No optimizer updates on real data and no Git writes (other than explicit fetch
commands supplied by the caller). Never stage/commit/push on a failed review.
"""
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
STATE_PATH = ROOT / '.local/bert_crf/stage1d_review_state.json'


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()


def prepare():
    if STATE_PATH.exists():
        raise FileExistsError(STATE_PATH)
    prior = json.loads((ROOT / '.local/bert_crf/stage1d1_state.json').read_text(encoding='utf-8'))
    stamp = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z') + '_stage1d_closeout_review'
    state = {**prior, 'run_id': stamp, 'started_at': datetime.now().astimezone().isoformat(),
             'evidence': str(ROOT / 'docs/bert_crf/evidence' / stamp),
             'local': str(ROOT / '.local/bert_crf' / stamp), 'prior_stage1d1_state': str(ROOT / '.local/bert_crf/stage1d1_state.json')}
    for key in ('evidence', 'local'):
        Path(state[key]).mkdir(parents=True, exist_ok=False)
    ev = Path(state['evidence'])
    files = set(p for p in git('ls-files', '-z').split('\0') if p)
    files.update(p for p in git('ls-files', '--others', '--exclude-standard', '-z').split('\0') if p)
    write(ev / 'initial_files.json', {p: digest(ROOT / p) for p in sorted(files)})
    write(ev / 'initial_git.json', {'branch': git('branch', '--show-current'), 'commit': git('rev-parse', 'HEAD'),
        'remote_address': git('remote', 'get-url', 'origin'), 'remote_tracking_commit': git('rev-parse', 'origin/dxc'),
        'divergence': git('rev-list', '--left-right', '--count', 'HEAD...origin/dxc'), 'status': git('status', '--short')})
    write(STATE_PATH, state)
    write(ev / 'run_context.json', state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def run(name, command):
    state = json.loads(STATE_PATH.read_text(encoding='utf-8'))
    ev, local = Path(state['evidence']), Path(state['local'])
    logs = local / 'logs'
    logs.mkdir(exist_ok=True)
    log = logs / (name + '.log')
    if log.exists():
        raise FileExistsError(log)
    overrides = {'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONPATH': os.pathsep.join([str(ROOT / 'src'), str(Path(state['interface']) / 'code')]),
        'IMCS_DATA_DIR': state['data'], 'BERT_MODEL_DIR': state['tokenizer'],
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'STAGE1D1_STATE': str(STATE_PATH)}
    env = {**os.environ, **overrides}
    started = datetime.now().astimezone().isoformat()
    tick = time.perf_counter()
    sources = [*ROOT.glob('src/bert_crf/*.py'), *ROOT.glob('tests/bert_crf/*.py'),
               *ROOT.glob('configs/bert_crf/*.json'), ROOT / 'docs/bert_crf/stage1d1_integration_check.py', Path(__file__)]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    with log.open('w', encoding='utf-8') as f:
        p = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    record = {'name': name, 'command': command, 'cwd': str(ROOT), 'environment_overrides': overrides,
        'started_at': started, 'elapsed_seconds': time.perf_counter() - tick, 'exit_code': p.returncode,
        'log': str(log), 'log_sha256': digest(log), 'source_hashes': source_hashes}
    with (ev / 'commands.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'source_hashes'}, ensure_ascii=False, indent=2))
    print(log.read_text(encoding='utf-8')[-6000:])
    return p.returncode


def probes():
    from dataclasses import asdict
    from types import SimpleNamespace
    from bert_crf.data_adapter import V2BatchProvider
    from bert_crf.experiment import environment
    from bert_crf.trainer import Trainer
    from bert_crf.training_config import TrainingConfig
    state = json.loads(STATE_PATH.read_text(encoding='utf-8'))
    ev, local = Path(state['evidence']), Path(state['local'])
    write(ev / 'review_environment.json', environment())
    prior = json.loads(Path(state['prior_stage1d1_state']).read_text(encoding='utf-8'))
    prior_identity = json.loads((Path(prior['evidence']) / 'input_identity.json').read_text(encoding='utf-8'))
    current_identity = {p: digest(p) for p in prior_identity}
    assert current_identity == prior_identity, 'Input identity changed since integration acceptance'
    write(ev / 'input_identity.json', current_identity)
    records = []
    forbidden = local / 'must_not_create_training_run'
    config = TrainingConfig(artifact_kind='real_training')
    try:
        Trainer(None, config, None, None, forbidden)
    except RuntimeError as e:
        records.append({'check': 'Trainer_real_training', 'result': 'REJECTED', 'message': str(e)})
    else:
        raise AssertionError('Expected synthetic-only gate changed; review scope must be reassessed')
    try:
        V2BatchProvider(SimpleNamespace(), None, provenance={'artifact_kind': 'real_training'})
    except ValueError as e:
        records.append({'check': 'Provider_real_training', 'result': 'REJECTED', 'message': str(e)})
    else:
        raise AssertionError('Unexpected real provider support; inspect before concluding')
    assert not forbidden.exists()
    # This is a negative configuration probe, explicitly NOT a training plan.
    write(local / 'negative_probe_real_training.json', asdict(config))
    write(ev / 'readiness_probes.json', {'probe_kind': 'unit_test_negative_probe', 'checks': records,
        'training_run_created': False, 'optimizer_created': False,
        'formal_config_files': [str(p.relative_to(ROOT)) for p in ROOT.glob('configs/bert_crf/*.json')
            if json.loads(p.read_text(encoding='utf-8')).get('artifact_kind') == 'real_training'],
        'conclusion': 'BLOCKED: real training cannot run by authorization/config alone; code integration is required'})
    print(json.dumps(records, ensure_ascii=False, indent=2))


def inventory():
    import re
    state = json.loads(STATE_PATH.read_text(encoding='utf-8'))
    ev = Path(state['evidence'])
    prior = json.loads(Path(state['prior_stage1d1_state']).read_text(encoding='utf-8'))
    previous = Path(prior['evidence'])
    protected = json.loads((previous / 'protected_manifest.json').read_text(encoding='utf-8'))
    assert all((ROOT / p).is_file() and digest(ROOT / p) == h for p, h in protected.items())
    prior_sources = json.loads((previous / 'source_manifest.json').read_text(encoding='utf-8'))
    changed_prior = [p for p, h in prior_sources.items() if digest(ROOT / p) != h]
    assert changed_prior == ['docs/bert_crf/05_character_adapter_integration.md'], changed_prior
    command_integrity = []
    roots = [Path(json.loads((ROOT / '.local/bert_crf/interface_v2_state.json').read_text(encoding='utf-8'))['evidence']), previous]
    for directory in roots:
        records = [json.loads(line) for line in (directory / 'commands.jsonl').read_text(encoding='utf-8').splitlines()]
        for rec in records:
            raw_log = rec.get('log', rec.get('raw_log_local_only'))
            assert digest(raw_log) == rec['log_sha256']
        command_integrity.append({'directory': str(directory), 'commands_verified': len(records)})
    materials = ['docs/bert_crf/' + name for name in ('00_initial_audit.md', '01_environment_and_dataloader.md',
        'alignment_summary.md', '02_model_core_and_smoke_tests.md', '03_training_framework.md',
        '04_interface_v2_audit.md', '05_character_adapter_integration.md')]
    materials += ['预测结果交接规范.md', '课程项目任务书_20261013_codex.md',
                  '项目报告模板_20261013_codex.md', '贡献声明表_20261013_codex.md']
    material_identity = {p: {'sha256': digest(ROOT / p), 'utf8_characters': len((ROOT / p).read_text(encoding='utf-8'))} for p in materials}
    files = [p for p in git('ls-files', '--others', '--exclude-standard', '-z').split('\0') if p]
    # Examine prospective public files, without staging them or exporting raw logs.
    data = json.loads((Path(state['data']) / 'IMCS-V2_train.json').read_text(encoding='utf-8'))
    needles = [(d, s['sentence_id'], s['sentence']) for d, dialog in data.items() for s in dialog['dialogue'] if len(s['sentence']) >= 8]
    findings, candidates = [], []
    for name in files:
        p = ROOT / name
        assert p.suffix in ('.md', '.py', '.json') and p.stat().st_size < 1024 * 1024
        text = p.read_text(encoding='utf-8')
        secrets = [kind for kind, pattern in {
            'private_key': r'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----',
            'github_token': r'gh[pousr]_[A-Za-z0-9]{30,}',
            'aws_access_key': r'AKIA[A-Z0-9]{16}'}.items() if re.search(pattern, text)]
        medical_matches = [{'dialogue_id': d, 'sentence_id': s, 'length': len(sentence),
            'sha256': hashlib.sha256(sentence.encode('utf-8')).hexdigest()} for d, s, sentence in needles if sentence in text]
        users = bool(re.search(r'[A-Za-z]:[/\\]+Users[/\\]', text, re.I))
        result = {'file': name, 'bytes': p.stat().st_size, 'sha256': digest(p),
            'secret_patterns': secrets, 'source_sentence_matches_ge8': medical_matches,
            'user_profile_absolute_path': users,
            'project_absolute_path': 'C:/userProgram/NLPexp' in text or 'C:\\userProgram\\NLPexp' in text}
        candidates.append(result)
        if secrets or medical_matches or users:
            findings.append(result)
    ignored = [Path(state['archive']), Path(state['data']), Path(state['pretrained']),
        ROOT / '.venv-bert-crf', roots[0], previous, ev, Path(state['local'])]
    ignored_checks = []
    for p in ignored:
        r = subprocess.run(['git', 'check-ignore', str(p.relative_to(ROOT))], cwd=ROOT, capture_output=True, encoding='utf-8')
        ignored_checks.append({'path': str(p.relative_to(ROOT)), 'ignored': r.returncode == 0})
    assert all(x['ignored'] for x in ignored_checks)
    write(ev / 'file_and_evidence_review.json', {'protected_unchanged': len(protected), 'changed_prior_sources': changed_prior,
        'prior_command_logs_verified': command_integrity, 'materials': material_identity,
        'candidates': candidates, 'sensitive_findings_requiring_review': findings,
        'automated_scan_scope': 'Secret formats, user-profile paths and exact original train sentences >=8 code points; manual source review complements this limited scan',
        'ignored_artifacts': ignored_checks, 'staged_files': git('diff', '--cached', '--name-only'),
        'git_status': git('status', '--short'), 'git_diff_check': subprocess.run(['git', 'diff', '--check'], cwd=ROOT).returncode})
    print(json.dumps({'protected_files_unchanged': len(protected), 'candidate_files': len(candidates),
        'sensitive_findings_requiring_review': len(findings), 'prior_command_log_integrity': command_integrity}, ensure_ascii=False, indent=2))


def closure():
    """Check the actual artifacts; a successful audit can conclude NOT READY."""
    import xml.etree.ElementTree as ET
    state = json.loads(STATE_PATH.read_text(encoding='utf-8'))
    ev = Path(state['evidence'])
    output = ev / 'review_closure.json'
    if output.exists():
        raise FileExistsError(output)
    read = lambda name: json.loads((ev / name).read_text(encoding='utf-8'))
    cases = ET.parse(ev / 'full_regression.xml').findall('.//testcase')
    assert len(cases) == 188
    assert not any(c.find(tag) is not None for c in cases for tag in ('failure', 'error', 'skipped'))
    new_cases = [c for c in cases if c.attrib['classname'].rsplit('.', 1)[-1]
                 in ('test_character_adapter', 'test_character_real_integration')]
    assert len(new_cases) == 58
    coverage = read('coverage_audit.json')
    counts = coverage['total']
    assert coverage['status'] == 'PASS' and counts['sentences'] == 98529 and counts['characters'] == 1279361
    assert counts['characters_covered_exactly_once'] == counts['characters']
    assert counts['gold_unchanged_sentences'] == counts['sentences']
    for key in ('coverage_gap_characters', 'coverage_overlap_characters', 'invalid_offset_sentences',
                'gold_length_mismatch_sentences', 'illegal_gold_bio_sentences', 'rejected_sentences'):
        assert counts[key] == 0
    real = read('real_batch_integration.json')
    assert real['status'] == 'PASS' and real['optimizer_steps'] == 0 and not real['optimizer_created']
    assert real['parameter_hash_before'] == real['parameter_hash_after']
    assert real['parameter_version_counters_unchanged'] and real['gold_free_decode_identical']
    for item in real['batches']['train']['gradients'].values():
        assert item['trainable_tensors'] == item['finite_gradient_tensors'] == item['nonzero_gradient_tensors']
    for item in real['batches'].values():
        assert item['prediction_lengths'] == item['character_lengths']
    assert real['batches']['val']['strict_bio_legal']
    assert all(real['checkpoint'][k] for k in ('parameter_hash_equal', 'eval_emissions_bitwise_equal', 'paths_equal'))
    eight = read('eight_cases_model.json')
    assert len(eight) == 8 and all(c['raw_gold_unchanged'] and c['loss_finite']
        and c['strict_bio_legal'] and c['prediction_length'] == c['n_char'] for c in eight)
    probes_result = read('readiness_probes.json')
    assert not probes_result['formal_config_files'] and not probes_result['training_run_created']
    assert all(c['result'] == 'REJECTED' for c in probes_result['checks'])
    commands = [json.loads(line) for line in (ev / 'commands.jsonl').read_text(encoding='utf-8').splitlines()]
    for record in commands:
        assert digest(record['log']) == record['log_sha256']
        assert record['exit_code'] == (1 if record['name'] == 'formal_entry_rejection' else 0)
    inspection = read('file_and_evidence_review.json')
    assert not inspection['sensitive_findings_requiring_review'] and inspection['git_diff_check'] == 0
    # Finalize only the report and this artifact checker after inventory.
    changed = [c['file'] for c in inspection['candidates'] if digest(ROOT / c['file']) != c['sha256']]
    assert set(changed) == {'docs/bert_crf/06_pretraining_readiness_review.md',
                            'docs/bert_crf/stage1d_closeout_review.py'}, changed
    candidates = [p for p in git('ls-files', '--others', '--exclude-standard', '-z').split('\0') if p]
    assert set(candidates) == {c['file'] for c in inspection['candidates']} and len(candidates) == 15
    assert git('diff', '--name-only') == '' and git('diff', '--cached', '--name-only') == ''
    diff_check = subprocess.run(['git', 'diff', '--check'], cwd=ROOT).returncode
    assert diff_check == 0
    baseline = read('initial_git.json')
    assert git('rev-parse', 'HEAD') == baseline['commit'] and git('branch', '--show-current') == 'dxc'
    remote_line = git('ls-remote', 'origin', 'refs/heads/dxc')
    assert remote_line.split()[0] == git('rev-parse', 'HEAD') == git('rev-parse', 'origin/dxc')
    summary = {
        'artifact_kind': 'readiness_review', 'completed_at': datetime.now().astimezone().isoformat(),
        'audit_execution': 'PASS', 'pretraining_readiness': 'FAIL', 'git_publish_authorized_by_review': False,
        'blockers': {'R1': 'Real Trainer/Provider modes and formal CLI are not wired',
                     'R2': 'Independent real training configuration and startup validation are absent'},
        'tests': {'passed': len(cases), 'failed': 0, 'skipped': 0, 'prior': 130, 'stage1d1': len(new_cases)},
        'coverage': counts, 'eight_cases_verified': len(eight),
        'real_optimizer_steps': 0, 'checkpoint_reload': 'PASS',
        'raw_command_logs_verified_before_closure': len(commands),
        'commands_manifest_sha256_before_closure_append': digest(ev / 'commands.jsonl'),
        'protected_files_unchanged': inspection['protected_unchanged'],
        'inventory_followup_change': {'files': changed, 'reason': 'Finalize report and read-only artifact verification'},
        'candidate_source_hashes': {p: digest(ROOT / p) for p in sorted(candidates)},
        'evidence_hashes': {p.name: digest(p) for p in sorted(ev.iterdir()) if p.is_file()},
        'git': {'branch': 'dxc', 'head': git('rev-parse', 'HEAD'), 'remote_actual': remote_line,
                'divergence': git('rev-list', '--left-right', '--count', 'HEAD...origin/dxc'),
                'status': git('status', '--short'), 'staged_files': [], 'tracked_changes': [],
                'untracked_files': candidates, 'diff_check_exit_code': diff_check,
                'new_commit_created': False, 'push_performed': False},
        'scope_limits': ['No formal training or official dev/test use',
                         'Privacy scan is limited and complemented by manual source review',
                         'Formal training route needs a separate implementation and acceptance task']}
    write(output, summary)
    print(json.dumps({k: summary[k] for k in ('audit_execution', 'pretraining_readiness', 'blockers', 'tests', 'git')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'probes', 'inventory', 'closure'])
    parser.add_argument('name', nargs='?')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'probes':
        probes()
    elif args.action == 'inventory':
        inventory()
    elif args.action == 'closure':
        closure()
    else:
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        sys.exit(run(args.name, command))
