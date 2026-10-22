"""Serializable model/runtime configuration, independent of dataset processing."""
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from .constraints import LABELS


@dataclass(frozen=True)
class ModelConfig:
    schema_version: str = '1.0'
    artifact_kind: str = 'stage1b_core_smoke_test'
    bert_model_name: str = 'google-bert/bert-base-chinese'
    bert_revision: str = '8f23c25b06e129b6c986331a13d8d025a92cf0ea'
    hidden_size: int = 768
    num_labels: int = 11
    labels: tuple[str, ...] = LABELS
    dropout: float = 0.1
    bio_constraints: bool = True
    constraint_rule_version: str = 'strict_bio_v1'
    seed: int = 1729
    device: str = 'cuda'
    precision: str = 'float32'
    deterministic_algorithms: bool = True

    def __post_init__(self):
        object.__setattr__(self, 'labels', tuple(self.labels))
        if self.labels != LABELS or self.num_labels != len(LABELS):
            raise ValueError('Only the fixed ordered 11-label IMCS BIO vocabulary is supported')
        if self.hidden_size <= 0 or not 0 <= self.dropout < 1:
            raise ValueError('Invalid hidden_size or dropout')
        if self.constraint_rule_version != 'strict_bio_v1' or self.schema_version != self._schema():
            raise ValueError('Unsupported config/rule version')
        if not isinstance(self.bio_constraints, bool) or not isinstance(self.deterministic_algorithms, bool):
            raise ValueError('Constraint/determinism flags must be bool')
        if self.precision != 'float32':
            raise ValueError('Stage 1B model runtime supports audited float32 only')
        if torch.device(self.device).type not in ('cpu', 'cuda'):
            raise ValueError('Only CPU/CUDA runtime supported')

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        return model_config_from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @staticmethod
    def _schema():
        return '1.0'


@dataclass(frozen=True)
class BoundaryAuxConfig:
    enabled: bool = True
    weight: float = 0.2
    target: str = 'entity_start_end'
    loss: str = 'balanced_bce_with_logits'
    decode_policy: str = 'crf_only'
    implementation_version: str = 'boundary_aux_v1'

    def __post_init__(self):
        expected = (True, 0.2, 'entity_start_end', 'balanced_bce_with_logits', 'crf_only', 'boundary_aux_v1')
        actual = (self.enabled, self.weight, self.target, self.loss, self.decode_policy, self.implementation_version)
        if type(self.enabled) is not bool or type(self.weight) not in (int, float) or actual != expected:
            raise ValueError('Unsupported boundary auxiliary design; fixed enabled/weight0.2/balanced BCE/CRF-only')


@dataclass(frozen=True)
class BoundaryModelConfig(ModelConfig):
    """Separate version: legacy ModelConfig serialization/identity stays unchanged."""
    schema_version: str = '1.1_boundary_aux'
    boundary_aux: BoundaryAuxConfig = field(default_factory=BoundaryAuxConfig)

    def __post_init__(self):
        if isinstance(self.boundary_aux, dict):
            object.__setattr__(self, 'boundary_aux', BoundaryAuxConfig(**self.boundary_aux))
        if not isinstance(self.boundary_aux, BoundaryAuxConfig):
            raise ValueError('Explicit boundary auxiliary configuration required')
        super().__post_init__()

    @staticmethod
    def _schema():
        return '1.1_boundary_aux'


def model_config_from_dict(value):
    cls = BoundaryModelConfig if value.get('schema_version') == '1.1_boundary_aux' else ModelConfig
    return cls(**value)


def configure_runtime(config: ModelConfig):
    """Explicitly apply the recorded seed/device/precision policy; no hidden global seeding."""
    if config.device.startswith('cuda') and config.deterministic_algorithms:
        setting = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if setting not in (':4096:8', ':16:8'):
            raise ValueError('Unsupported deterministic CUBLAS_WORKSPACE_CONFIG')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(config.deterministic_algorithms)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if config.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device unavailable')
    return torch.device(config.device)
