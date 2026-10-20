"""Local internal-validation artifacts, never official dev/test handoff output."""
import json
from pathlib import Path
import shutil

from .constraints import LABELS
from .experiment import canonical, file_hash, write_json
from .metrics import checked_tags, entities


class ValidationArtifacts:
    def __init__(self, run, epoch, kind):
        self.path = Path(run) / 'validation' / f'epoch_{epoch:04d}'
        self.path.mkdir(parents=True, exist_ok=False)
        self.kind = kind
        self.count = 0
        self.streams = {name: (self.path / (name + '.jsonl.partial')).open('x', encoding='utf-8')
                        for name in ('predictions', 'errors')}

    def add(self, key, gold, prediction):
        tags = checked_tags(prediction)
        if len(tags) != len(gold):
            raise ValueError('Validation artifact character length mismatch')
        record = {'artifact_kind': self.kind, 'split': 'internal_val', 'dialogue_id': key[0],
                  'sentence_id': key[1], 'character_count': len(tags), 'labels': tags}
        self.streams['predictions'].write(canonical(record) + '\n')
        g, p = entities(gold), entities(prediction)
        if g != p:
            error = {k: v for k, v in record.items() if k != 'labels'}
            error.update(missed_entities=sorted(g - p), spurious_entities=sorted(p - g))
            self.streams['errors'].write(canonical(error) + '\n')
        self.count += 1

    def finish(self, metrics, elapsed):
        for name, stream in self.streams.items():
            stream.close()
            (self.path / (name + '.jsonl.partial')).replace(self.path / (name + '.jsonl'))
        report = {'artifact_kind': self.kind, 'split': 'internal_val', 'sentences': self.count,
            'metrics': metrics, 'elapsed_seconds': elapsed,
            'timing_scope': 'internal validation loader/adaptation, GPU forward/decode, metrics and JSONL writes; excludes model loading and optimizer updates',
            'predictions_sha256': file_hash(self.path / 'predictions.jsonl'),
            'errors_sha256': file_hash(self.path / 'errors.jsonl')}
        write_json(self.path / 'validation_info.json', report)
        return report

    def abort(self):
        for stream in self.streams.values():
            stream.close()
        # Partial files are evidence of failure; never publish them as complete.
        write_json(self.path / 'failure.json', {'status': 'incomplete', 'sentences_written': self.count})


def initialize_run_info(trainer):
    update_run_info(trainer, 'created')


def update_run_info(trainer, status):
    context = trainer.execution_context
    record = {'schema_version': 'bert_crf_internal_run_v1', 'artifact_kind': trainer.config.artifact_kind,
        'run_id': trainer.run.manifest['run_id'], 'status': status,
        'method': 'character_bert_crf', 'code': trainer.run.manifest['code'],
        'config_file': 'config.json', 'config_sha256': file_hash(trainer.run.path / 'config.json'),
        'environment_file': 'environment.json', 'random_seed': trainer.config.seed,
        'started_at': trainer.run.manifest['started_at'],
        'updated_at': trainer.run.manifest.get('updated_at'),
        'ended_at': trainer.run.manifest.get('updated_at') if status in ('completed', 'early_stopped', 'failed', 'interrupted') else None,
        'fit_elapsed_seconds': trainer.run.manifest.get('elapsed_seconds'),
        'data': {'train': trainer.train_provider.metadata, 'validation': trainer.val_provider.metadata},
        'training_state': trainer.state, 'last_validation': trainer._last_validation_artifact,
        'selected_validation': getattr(trainer, '_selected_validation', None),
        'executor': context.config.value['executor'] if context is not None else 'synthetic_test',
        'training_command': trainer.run.manifest['command'],
        'alignment': context.config.value['alignment'] if context is not None else 'synthetic fixture',
        'official_handoff': {'status': 'not_run', 'schema_version': '1.0',
                             'reason': 'Internal validation artifacts are not official dev/test predictions'}}
    write_json(trainer.run.path / 'run_info.json', record)


def finalize_artifacts(trainer, status):
    state = trainer.state
    if state['best_checkpoint']:
        source_run = Path(state['best_checkpoint']).parent.parent
        source = source_run / 'validation' / f"epoch_{state['best_epoch']:04d}"
        info = json.loads((source / 'validation_info.json').read_text(encoding='utf-8'))
        trainer._selected_validation = info
        for name in ('predictions', 'errors'):
            p = source / (name + '.jsonl')
            if file_hash(p) != info[name + '_sha256']:
                raise ValueError('Best validation artifact hash mismatch')
            target = trainer.run.path / (name + '.jsonl')
            if target.exists():
                raise FileExistsError(target)
            shutil.copyfile(p, target)
        write_json(trainer.run.path / 'metrics.json', {'artifact_kind': trainer.config.artifact_kind,
            'best_epoch': state['best_epoch'], 'best_metric': state['best_value'],
            'validation': info['metrics'], 'validation_source_run': str(source_run),
            'validation_elapsed_seconds': info['elapsed_seconds'], 'timing_scope': info['timing_scope']})
    update_run_info(trainer, status)
