"""Strict, explicit formal-run configuration. No synthetic defaults or fallback."""
from dataclasses import fields
import json
import math
from pathlib import Path
import re

from .character_mapping import MAPPING_POLICY, OFFSET_UNIT
from .config import ModelConfig
from .constraints import LABELS
from .experiment import fingerprint
from .training_config import TrainingConfig


def exact(value, keys, where):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{where}: missing or unknown configuration fields')


class FormalConfig:
    def __init__(self, value, *, root):
        self.root = Path(root).resolve()
        # JSON roundtrip isolates the caller's mutable dictionary and rejects NaN.
        self.value = json.loads(json.dumps(value, allow_nan=False))
        v = self.value
        exact(v, {'schema_version', 'artifact_kind', 'experiment_name', 'executor', 'model', 'training',
            'inputs', 'data', 'alignment', 'resources', 'outputs', 'formats', 'warmup_ratio'}, 'formal')
        if v['schema_version'] != 'bert_crf_formal_v1' or v['artifact_kind'] != 'real_training':
            raise ValueError('Explicit formal real_training configuration required')
        for key in ('experiment_name', 'executor'):
            if not isinstance(v[key], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', v[key]):
                raise ValueError(f'Invalid {key}')
        exact(v['model'], {f.name for f in fields(ModelConfig)}, 'model')
        exact(v['training'], {f.name for f in fields(TrainingConfig)}, 'training')
        self.model = ModelConfig(**v['model'])
        self.training = TrainingConfig(**v['training'])
        if (self.model.bert_model_name, self.model.bert_revision) != ('google-bert/bert-base-chinese',
                '8f23c25b06e129b6c986331a13d8d025a92cf0ea'):
            raise ValueError('Baseline requires the audited model/revision')
        if self.model.artifact_kind != 'real_training' or self.training.artifact_kind != 'real_training':
            raise ValueError('Formal model/training artifact_kind mismatch')
        if (self.model.seed, self.model.device, self.model.precision, self.model.deterministic_algorithms) != \
                (self.training.seed, self.training.device, self.training.precision, self.training.deterministic_algorithms):
            raise ValueError('Model/training runtime mismatch')
        if not self.model.bio_constraints or tuple(self.model.labels) != LABELS:
            raise ValueError('Strict fixed BIO constraints required')
        if self.training.selection_metric != 'entity.micro.f1' or self.training.selection_mode != 'max' or self.training.min_delta != 0:
            raise ValueError('Baseline requires strict entity micro F1/max; retain earlier ties')
        if self.training.model_config_path != 'inline:formal.model':
            raise ValueError('Formal model configuration must be explicit inline')
        ratio = v['warmup_ratio']
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio < 1:
            raise ValueError('Invalid warmup_ratio')
        exact(v['data'], {'version', 'split_version', 'source', 'train_split', 'validation_split',
                          'expected_dialogues', 'expected_sentences', 'expected_characters'}, 'data')
        if (v['data']['source'], v['data']['train_split'], v['data']['validation_split']) != \
                ('official_train_only', 'train', 'val'):
            raise ValueError('Only fixed internal train/val from official train permitted')
        for key in ('version', 'split_version'):
            if not isinstance(v['data'][key], str) or not v['data'][key]:
                raise ValueError('Data/split version required')
        for key in ('expected_dialogues', 'expected_sentences', 'expected_characters'):
            exact(v['data'][key], {'train', 'val'}, key)
            if any(type(x) is not int or x <= 0 for x in v['data'][key].values()):
                raise ValueError('Positive split counts required')
        expected_counts = {'expected_dialogues': {'train': 2225, 'val': 247},
            'expected_sentences': {'train': 88842, 'val': 9687},
            'expected_characters': {'train': 1153312, 'val': 126049}}
        if any(v['data'][key] != counts for key, counts in expected_counts.items()):
            raise ValueError('Baseline split counts differ from audited data')
        planned = math.ceil(v['data']['expected_sentences']['train'] / self.training.batch_size) * self.training.epochs
        if self.training.warmup_steps != math.ceil(planned * ratio):
            raise ValueError('warmup_steps does not match ratio/planned optimizer steps')
        if v['alignment'] != {'policy': MAPPING_POLICY, 'offset_unit': OFFSET_UNIT, 'gold': 'original_character_bio',
                              'uncovered': 'reject', 'overlap': 'reject', 'truncation': 'reject'}:
            raise ValueError('Unsupported alignment/gold/exception policy')
        exact(v['resources'], {'max_tokens', 'padding', 'num_workers', 'gradient_accumulation_steps',
            'amp', 'adam_foreach', 'optimizer_state_reserve_multiplier', 'minimum_free_bytes',
            'minimum_disk_free_bytes', 'measurement_file', 'measurement_sha256'}, 'resources')
        r = v['resources']
        if type(r['max_tokens']) is not int or not 3 <= r['max_tokens'] <= 512:
            raise ValueError('Invalid max_tokens')
        if r['padding'] != 'dynamic_longest' or type(r['num_workers']) is not int or r['num_workers'] != 0:
            raise ValueError('Audited dynamic padding/num_workers=0 required')
        if type(r['gradient_accumulation_steps']) is not int or r['gradient_accumulation_steps'] != 1:
            raise ValueError('Baseline supports accumulation=1 only')
        if r['amp'] is not False or r['adam_foreach'] is not False or r['optimizer_state_reserve_multiplier'] != 3:
            raise ValueError('Audited FP32 AdamW foreach=False and moment/temporary reserve required')
        for key in ('minimum_free_bytes', 'minimum_disk_free_bytes'):
            if type(r[key]) is not int or r[key] <= 0:
                raise ValueError(f'Invalid {key}')
        self.path(r['measurement_file'])
        self._hash(r['measurement_sha256'])
        required = {'official_train', 'fixed_split', 'interface', 'tokenizer', 'pretrained'}
        exact(v['inputs'], required, 'inputs')
        for key, entry in v['inputs'].items():
            exact(entry, {'path', 'files'}, f'inputs.{key}')
            self.path(entry['path'])
            if not isinstance(entry['files'], dict) or not entry['files']:
                raise ValueError('Input file hashes required')
            for name, checksum in entry['files'].items():
                if name != '.' and (Path(name).is_absolute() or '..' in Path(name).parts):
                    raise ValueError('Unsafe input manifest member')
                self._hash(checksum)
        pinned = {'official_train': ('.', '07cdd3799087283a5a17de8bb43e71cb159a9a2c99c1cfbe8081847fac358f24'),
            'fixed_split': ('.', '497f21cf13d8f5accb83f86ffda16a4a59ba716c064ec347e990a219e8957072'),
            'tokenizer': ('tokenizer.json', '55c5edcfb8936895e019b2d8f97dcb547734f25b29f95009155f4c8c15649aae'),
            'pretrained': ('model.safetensors', '3404a1ffd8da507042e8161013ba2a4fc49858b4e3f8fbf5ce5724f94883aec3')}
        for kind, (member, checksum) in pinned.items():
            if v['inputs'][kind]['files'].get(member) != checksum:
                raise ValueError(f'Baseline identity differs from audited Stage 1D: {kind}')
        if v['formats'] != {'history': 'bert_crf_history_v1', 'predictions': 'character_internal_jsonl_v1',
                            'run_info': 'bert_crf_internal_run_v1', 'future_handoff': '1.0'}:
            raise ValueError('Unsupported artifact format')
        exact(v['outputs'], {'root', 'checkpoints', 'logs', 'results'}, 'outputs')
        if (v['outputs']['checkpoints'], v['outputs']['logs'], v['outputs']['results']) != ('checkpoints', '.', '.'):
            raise ValueError('Outputs must live inside one unique experiment directory')
        output = self.path(v['outputs']['root'])
        if output != self.root / 'experiments/bert_crf':
            raise ValueError('Formal output root must be experiments/bert_crf')

    @staticmethod
    def _hash(value):
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('Expected SHA-256')

    def path(self, value):
        if not isinstance(value, str) or not value or Path(value).is_absolute() or '..' in Path(value).parts:
            raise ValueError('Configuration paths must be project-relative without traversal')
        p = (self.root / value).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError('Path escapes project root')
        return p

    @property
    def identity(self):
        return fingerprint(self.value)

    @classmethod
    def load(cls, path, *, root):
        def unique(pairs):
            result = {}
            for key, val in pairs:
                if key in result:
                    raise ValueError('Duplicate configuration key')
                result[key] = val
            return result
        return cls(json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique), root=root)
