"""Copy an explicit allowlist of existing evidence; never rerun experiments."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'docs/bert_crf/published_evidence'
EVIDENCE = ROOT / 'docs/bert_crf/evidence'
STAGE0 = '20261019T160537+0800'
STAGE1A = '20261019T162653+0800_stage1a'
STAGE1B = '20261019T174356+0800_stage1b'

ALLOW = {
    STAGE0: ['dataset_archive_manifest.json', 'environment.json', 'input_manifest.json',
             'interface_manifest.json', 'pure_alignment_probes.json', 'secondary_environment.json',
             'shell_failure_note.json', 'source_integrity_check.json', 'split_verification.json',
             'torch_wheel_index_check.json', 'train_contract_checks.json', 'upstream_manifest.json'],
    STAGE1A: ['alignment_audit.json', 'batch_checks.json', 'commands.jsonl', 'contract_checks.json',
              'execution_summary.json', 'gpu_smoke.json', 'input_identity.json',
              'installation_plan.json', 'installation_sources.json', 'interface_manifest.json',
              'requirements-stage1a.in', 'run_context.json', 'tokenizer_manifest.json'],
    STAGE1B: ['acceptance.json', 'acceptance.log', 'commands.jsonl', 'complete_tests.log',
              'complete_tests.xml', 'complete_tests_reporting_fix.log',
              'complete_tests_reporting_fix.xml', 'core_tests_first.log', 'core_tests_first.xml',
              'dependency_check.log', 'formatting_record.json', 'freeze.log', 'git_commit.log',
              'git_final.json', 'git_final.log', 'git_whitespace_final.log',
              'git_whitespace_first.log', 'pretrained_download.log', 'pretrained_gpu_smoke.json',
              'pretrained_manifest.json', 'run_context.json', 'source_manifest.json',
              'source_manifest_tested.json', 'test_pretrained_before_reporting_fix.py'],
}
LOCAL_LOGS = ['00_create_venv.log', '03_cuda_smoke.log', '05_cuda_smoke_retry.log',
              '07_public_unittest.log', '08_full_alignment_audit.log', '09_pip_check.log',
              '11_freeze.log', '12_projected_gold_and_contract.log', '13_cuda_recheck.log']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    pairs = [(EVIDENCE / run / name, DEST / run / name)
             for run, names in ALLOW.items() for name in names]
    pairs += [(ROOT / '.local/bert_crf' / STAGE1A / 'logs' / name,
               DEST / STAGE1A / 'logs' / name) for name in LOCAL_LOGS]
    # Fail on obvious credentials. This supplements the explicit content review,
    # not a claim that regex can establish privacy for arbitrary medical logs.
    forbidden = re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|'
                           r'\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]+|'
                           r'\bhf_[A-Za-z0-9]{25,}|(?:X-Amz-Signature|X-Goog-Signature)=')
    records = []
    for source, target in pairs:
        data = source.read_bytes()
        if forbidden.search(data.decode('utf-8')):
            raise ValueError(f'Possible credential in allowlisted file: {source.name}')
        if target.exists():
            raise FileExistsError(f'Do not overwrite published evidence: {target}')
        records.append({'source': source.relative_to(ROOT).as_posix(),
                        'published': target.relative_to(ROOT).as_posix(),
                        'bytes': len(data), 'sha256': digest(data), 'transformation': 'none; byte-identical copy'})
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (DEST / 'manifest.json').write_text(json.dumps({
        'artifact_kind': 'published_development_evidence',
        'exported_at': datetime.now().astimezone().isoformat(),
        'experiment_times': 'Unmodified original timestamps in each run; export is not a new experiment',
        'excluded': ['medical original text and demo log', 'private alignment samples',
                     'datasets and split lists', 'model weights and checkpoints',
                     'environments and SSH keys', 'duplicated third-party source snapshots'],
        'files': records}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {len(records)} byte-identical files, {sum(r["bytes"] for r in records)} bytes')


if __name__ == '__main__':
    main()
