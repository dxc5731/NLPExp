"""Stage 1E real measurements only: no real optimizer or formal fit calls."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import time

import torch

from bert_crf.character_mapping import MAPPING_POLICY, OFFSET_UNIT
from bert_crf.config import ModelConfig
from bert_crf.experiment import environment, file_hash, write_json
from bert_crf.formal_config import FormalConfig
from bert_crf.preflight import build_model, probe_batch, scan_providers
from bert_crf.real_data import build_providers, sample_batch
from bert_crf.training_config import TrainingConfig

ROOT = Path(__file__).resolve().parents[2]
STATE = json.loads((ROOT / '.local/bert_crf/stage1e_state.json').read_text(encoding='utf-8'))
EV, LOCAL = Path(STATE['evidence']), Path(STATE['local'])
CONFIG = ROOT / 'configs/bert_crf/first_baseline.json'
PUBLIC = ROOT / 'docs/bert_crf/published_evidence' / STATE['run_id']


def manifest(path):
    path = Path(path)
    members = {'.': file_hash(path)} if path.is_file() else {
        p.relative_to(path).as_posix(): file_hash(p) for p in sorted(path.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    return {'path': path.relative_to(ROOT).as_posix(), 'files': members}


def configure():
    if CONFIG.exists():
        raise FileExistsError(CONFIG)
    model = replace(ModelConfig(), artifact_kind='real_training', seed=42)
    training = replace(TrainingConfig(), artifact_kind='real_training', epochs=5, batch_size=8, seed=42,
        early_stopping_patience=2, warmup_steps=math.ceil(math.ceil(88842 / 8) * 5 * 0.1),
        model_config_path='inline:formal.model')
    config = {'schema_version': 'bert_crf_formal_v1', 'artifact_kind': 'real_training',
        'experiment_name': 'first_baseline', 'executor': 'dxc', 'model': asdict(model), 'training': asdict(training),
        'inputs': {k: manifest(p) for k, p in {
            'official_train': Path(STATE['data']) / 'IMCS-V2_train.json',
            'fixed_split': Path(STATE['interface']) / 'code/artifacts/split_v1.json',
            'interface': Path(STATE['interface']) / 'code/src',
            'tokenizer': Path(STATE['tokenizer']), 'pretrained': Path(STATE['pretrained'])}.items()},
        'data': {'version': 'IMCS-V2-NER_train_sha256_07cdd379', 'split_version': 'split_v1_sha256_497f21cf',
            'source': 'official_train_only', 'train_split': 'train', 'validation_split': 'val',
            'expected_dialogues': {'train': 2225, 'val': 247}, 'expected_sentences': {'train': 88842, 'val': 9687},
            'expected_characters': {'train': 1153312, 'val': 126049}},
        'alignment': {'policy': MAPPING_POLICY, 'offset_unit': OFFSET_UNIT, 'gold': 'original_character_bio',
            'uncovered': 'reject', 'overlap': 'reject', 'truncation': 'reject'},
        'resources': {'max_tokens': 256, 'padding': 'dynamic_longest', 'num_workers': 0,
            'gradient_accumulation_steps': 1, 'amp': False, 'adam_foreach': False,
            'optimizer_state_reserve_multiplier': 3, 'minimum_free_bytes': 1073741824,
            'minimum_disk_free_bytes': 17179869184,
            'measurement_file': (PUBLIC / 'resource_measurements.json').relative_to(ROOT).as_posix(),
            'measurement_sha256': '0' * 64},
        'outputs': {'root': 'experiments/bert_crf', 'checkpoints': 'checkpoints', 'logs': '.', 'results': '.'},
        'formats': {'history': 'bert_crf_history_v1', 'predictions': 'character_internal_jsonl_v1',
            'run_info': 'bert_crf_internal_run_v1', 'future_handoff': '1.0'}, 'warmup_ratio': 0.1}
    FormalConfig(config, root=ROOT)
    write_json(CONFIG, config)
    write_json(EV / 'initial_candidate_config.json', config)
    print('Candidate saved; resource measurement remains required before startup')


def survey():
    config = FormalConfig.load(CONFIG, root=ROOT)
    providers, _ = build_providers(config)
    tick = time.perf_counter()
    stats = scan_providers(config, providers)
    output = {'artifact_kind': 'environment_check', 'status': 'PASS', 'splits': stats,
        'optimizer_steps': 0, 'elapsed_seconds': time.perf_counter() - tick,
        'inputs': config.value['inputs'], 'mapping_policy': MAPPING_POLICY}
    target = EV / 'lengths_and_coverage.json'
    if target.exists():
        raise FileExistsError(target)
    write_json(target, output)
    print(json.dumps({k: {m: v for m, v in x['token_lengths'].items() if m != 'histogram'}
                      for k, x in stats.items()}, indent=2))


def resources():
    config = FormalConfig.load(CONFIG, root=ROOT)
    providers, _ = build_providers(config)
    p = providers['train']
    model = build_model(config)
    lengths = [len(x['input_ids']) for x in p.adapter.dataset.encoded]
    longest = sorted(range(p.size), key=lambda i: lengths[i], reverse=True)
    ordinary = torch.randperm(p.size, generator=torch.Generator().manual_seed(config.training.seed + 1)).tolist()
    cases = []
    report = {'artifact_kind': 'real_data_resource_test', 'optimizer_steps': 0,
        'model_config': json.loads(json.dumps(asdict(config.model))), 'inputs': config.value['inputs'],
        'environment': environment(), 'cases': cases,
        'notes': ['No optimizer created or stepped', '3 parameter-sized FP32 tensors reserve two AdamW moments and working memory',
                  'Step timings exclude AdamW, checkpoint I/O and a full epoch; estimates are not measured epoch duration']}
    target = EV / 'resource_measurements.json'
    if target.exists():
        raise FileExistsError(target)
    # Small warmup graph, still without updates; exclude it from timing summary.
    probe_batch(model, p, ordinary[:2], backward=True)
    failed = False
    for batch_size in (4, 8, 16):
        for strategy, order in (('seeded_ordinary', ordinary), ('longest_unique', longest)):
            try:
                case = probe_batch(model, p, order[:batch_size], backward=True, reserve_multiplier=3)
                tick = time.perf_counter()
                for _ in range(3):
                    sample_batch(p, order[:batch_size])
                case['cpu_adaptation_seconds_mean_3'] = (time.perf_counter() - tick) / 3
            except torch.cuda.OutOfMemoryError as exc:
                case = {'status': 'OOM', 'batch_size': batch_size, 'error': str(exc), 'reserve_multiplier': 3}
                failed = True
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            case.update(sample_strategy=strategy, max_tokens=config.value['resources']['max_tokens'])
            cases.append(case)
            write_json(target, report)
            print(json.dumps(case), flush=True)
            if failed:
                break
        if failed:
            break
    write_json(target, report)
    print('Resource evidence saved; no formal training called')


def select():
    measurement = json.loads((EV / 'resource_measurements.json').read_text(encoding='utf-8'))
    safe = [c['batch_size'] for c in measurement['cases'] if c['status'] == 'PASS'
        and c['sample_strategy'] == 'longest_unique' and c['free_bytes_at_peak_end'] >= 1073741824]
    if not safe:
        raise RuntimeError('No resource-safe batch candidate')
    chosen = max(safe)
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    config['training']['batch_size'] = chosen
    config['training']['warmup_steps'] = math.ceil(math.ceil(88842 / chosen) * config['training']['epochs'] * config['warmup_ratio'])
    PUBLIC.mkdir(parents=True, exist_ok=True)
    published = PUBLIC / 'resource_measurements.json'
    if published.exists():
        raise FileExistsError(published)
    import shutil
    shutil.copyfile(EV / 'resource_measurements.json', published)
    config['resources']['measurement_sha256'] = file_hash(published)
    FormalConfig(config, root=ROOT)
    write_json(CONFIG, config)
    write_json(EV / 'selected_config.json', config)
    write_json(EV / 'configuration_selection.json', {'artifact_kind': 'environment_check', 'batch_size': chosen,
        'basis': 'Largest measured candidate with longest unique real samples, three-copy AdamW reserve and >=1GiB free',
        'formal_training_started': False, 'config_sha256': file_hash(CONFIG)})
    print('Resource-selected batch_size=' + str(chosen))


def synthetic_pretrained():
    from bert_crf.synthetic_smoke import pretrained_components
    from bert_crf.trainer import Trainer
    tick = time.perf_counter()
    config = replace(TrainingConfig(), epochs=2, seed=42, batch_size=2, early_stopping_patience=2)
    model, train, val = pretrained_components(config, STATE['pretrained'])
    trainer = Trainer(model, config, train, val, LOCAL / 'synthetic_runs',
        metadata={'purpose': 'stage1e_synthetic_pretrained_trainer_check', 'real_data_used': False},
        record_validation_artifacts=True)
    names = ('encoder.bert.embeddings.word_embeddings.weight', 'head.classifier.weight', 'head.crf.transitions')
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name in names}
    trainer.fit()
    changes = {name: float((p.detach().cpu() - before[name]).abs().max())
               for name, p in model.named_parameters() if name in before}
    assert all(value > 0 for value in changes.values())
    trainer.load_best()
    metrics = trainer.evaluate()
    output = {'artifact_kind': 'synthetic_smoke_test', 'status': 'PASS', 'real_data_used': False,
        'config': config.to_dict(), 'state': trainer.state, 'max_parameter_changes': changes,
        'best_reloaded_validation': metrics, 'run_directory': str(trainer.run.path),
        'elapsed_seconds_including_load': time.perf_counter() - tick,
        'manifest': json.loads((trainer.run.path / 'run_manifest.json').read_text(encoding='utf-8'))}
    write_json(EV / 'synthetic_pretrained_trainer.json', output)
    print(json.dumps({k: v for k, v in output.items() if k != 'manifest'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['configure', 'survey', 'resources', 'select', 'synthetic_pretrained'])
    globals()[parser.parse_args().action]()
