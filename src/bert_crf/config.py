"""Serializable model/runtime configuration, independent of dataset processing."""
from dataclasses import asdict, dataclass
import json
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
        if self.constraint_rule_version != 'strict_bio_v1' or self.schema_version != '1.0':
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
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


def configure_runtime(config: ModelConfig):
    """Explicitly apply the recorded seed/device/precision policy; no hidden global seeding."""
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
