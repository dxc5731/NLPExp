"""Targeted checks and end-of-audit integrity; no model/mapping/training code."""
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from interface_v2_runner import STATE, digest, dump


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    state = read(STATE)
    evidence, interface = Path(state['evidence']), Path(state['interface'])
    sys.path.insert(0, str(interface / 'code'))
    from src.align import encode_sentence, load_tokenizer, restore_char_labels
    from src.data_io import load_split
    from src.labels import LABELS
    from interface_v2_alignment_audit import bio_violations
    tokenizer = load_tokenizer()
    data = load_split('train')
    issues = [json.loads(s) for s in (evidence / 'alignment_issues.jsonl').read_text(encoding='utf-8').splitlines()]
    results = []
    for issue in issues:
        did, sid = issue['dialogue_id'], issue['sentence_id']
        source = next(s for s in data[did]['dialogue'] if s['sentence_id'] == sid)
        tags = source['BIO_label'].split(' ')
        encoded = encode_sentence(tokenizer, source['sentence'], tags=tags, where=f'{did}/{sid}')
        kept = [(i, LABELS[label]) for i, label in enumerate(encoded.token_labels) if label != -100]
        filtered = [tag for _, tag in kept]
        violations = bio_violations(filtered)
        for item in violations:
            item['original_token_index'] = kept[item['position']][0]
        restored = restore_char_labels(encoded.token_labels, encoded.offsets, len(tags))
        assert [i for i, t in enumerate(restored) if t is None] == issue['missing_restored_label_positions']
        results.append({'dialogue_id': did, 'sentence_id': sid, 'split': issue['split'],
                        'original_char_BIO_valid': not bio_violations(tags),
                        'filtered_token_BIO_violations': violations,
                        'note': 'Hypothetical token-CRF compaction only; no tensors mapped or model run'})
    assert sum(bool(x['filtered_token_BIO_violations']) for x in results) == 2
    declared = read(interface / 'code/artifacts/alignment_report_train.json')
    # No dev/test audit JSON is parsed; only file hashes are collected.
    package_report_summary = {k: v for k, v in declared.items() if isinstance(v, (int, float, bool))}
    dump(evidence / 'targeted_rechecks.json', {'results': results, 'v2_train_report_numeric_fields': package_report_summary,
         'ambiguous_intersecting_entities': sum(len(a['intersecting_gold_entities']) for x in issues for a in x['ambiguous'])})
    mapping = [
        ('input_ids', 'batch.input_ids', '[B,L] int64', 'reuse', 'actual batch'),
        ('attention_mask', 'batch.attention_mask', '[B,L] int64', 'reuse for BERT, not character CRF', 'actual batch'),
        ('character_labels', 'Dataset.char_labels indexed by original ID', '[B,C] int64, 0..10; O only as masked padding placeholder', 'NEW data adapter; never infer from token labels', 'source gold verified, adapter not implemented'),
        ('character_mask', 'len(batch.text)', '[B,C] bool contiguous prefix', 'NEW data adapter', 'lengths checked, tensor adapter not implemented'),
        ('mapping', 'batch.offsets + attention + original lengths', 'future mapping metadata', 'NEW data adapter AND differentiable representation adapter', 'coverage audited only; no representation conversion'),
        ('ids', 'batch.dialogue_id, batch.sentence_id', 'list[tuple[str,str]]', 'zip into Trainer protocol; preserve strings/order', 'actual IDs verified'),
        ('token_type_ids', 'not supplied', 'optional [B,L]', 'single sentence BERT can omit', 'static model signature'),
        ('character_hidden_states', 'BERT output plus future adapter, NOT DataLoader', '[B,C,H]', 'NEW differentiable representation adapter', 'not implemented / not run'),
        ('provider.metadata', 'archive/split/tokenizer/label/alignment identities', 'versioned dict', 'NEW epoch-pure BatchProvider wrapper', 'source identities available; wrapper not implemented'),
    ]
    dump(evidence / 'field_mapping.json', {'inspection_kind': 'static integration analysis backed by actual data batches',
        'fields': [dict(zip(('target', 'source', 'format', 'required_work', 'verification'), row)) for row in mapping],
        'real_execution_guard': 'TrainingConfig.require_synthetic and formal entry remain unchanged/closed',
        'best_baseline_pending_human_confirmation': 'Validated offsets; share one token vector across its span; mean only for explicitly allowed multi-token coverage; fail on zero coverage. No gold-dependent mapping.',
        'gradient_status': 'Gather/share/mean are differentiable in theory; no v2 gradient or model test was run'})
    before = read(evidence / 'initial_git.json')
    assert all(digest(ROOT / p) == h for p, h in before['tracked_hashes'].items())
    manifest = read(evidence / 'interface_manifest.json')
    assert all(digest(interface / m['path']) == m['sha256'] for m in manifest)
    old = read(ROOT / 'docs/bert_crf/evidence/20261019T162653+0800_stage1a/interface_manifest.json')
    old_root = ROOT / '.local/bert_crf/20261019T162653+0800_stage1a/interface'
    assert all(digest(old_root / m['path']) == m['sha256'] for m in old)
    identities = read(evidence / 'input_identity.json')
    assert digest(state['archive']) == identities['v2_archive_sha256']
    assert digest(ROOT / 'IMCS-V2-NER.zip') == identities['data_archive_sha256']
    assert digest(Path(state['data']) / 'IMCS-V2_train.json') == identities['train_source_sha256']
    for item in identities['tokenizer']['files']:
        assert digest(Path(state['tokenizer']) / item['name']) == item['sha256']
    commands = [json.loads(s) for s in (evidence / 'commands.jsonl').read_text(encoding='utf-8').splitlines()]
    for command in commands:
        assert digest(command['raw_log_local_only']) == command['log_sha256']
        assert command['exit_code'] == 0
    tests = Path(next(c['raw_log_local_only'] for c in commands if c['name'] == 'supplied_tests')).read_text(encoding='utf-8')
    assert re.search(r'Ran 13 tests in', tests) and re.search(r'\nOK\s*$', tests)
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()
    assert git('rev-parse', 'HEAD') == before['commit']
    source_paths = [*ROOT.glob('src/bert_crf/*.py'), *ROOT.glob('tests/bert_crf/*.py'),
                    *ROOT.glob('docs/bert_crf/interface_v2*.py')]
    dump(evidence / 'source_manifest.json', {p.relative_to(ROOT).as_posix(): digest(p) for p in source_paths})
    summary = {'finished_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'artifact_kind': 'interface_audit',
        'git': {'branch': git('branch', '--show-current'), 'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--porcelain')},
        'tracked_files_byte_identical': len(before['tracked_hashes']), 'v2_files_byte_identical': len(manifest),
        'v1_files_byte_identical': len(old), 'archive_data_tokenizer_hashes_unchanged': True,
        'supplied_tests': {'passed': 13, 'failed': 0}, 'prior_model_tests_rerun': False,
        'prior_model_tests_note': 'Historical 130 passes are not a new run; no model/optimizer executed in this audit.',
        'all_logged_commands_at_this_point_exit_zero': True, 'demo_roundtrip_true': next(c['demo_roundtrip_true'] for c in commands if c['name'] == 'demo'),
        'committed_or_pushed': False, 'real_adapter_implemented': False,
        'real_data_training': False, 'formal_predictions': False, 'dev_test_gold_used': False}
    dump(evidence / 'execution_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
