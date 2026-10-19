"""Validated training policy; synthetic configs can never start real training."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

KINDS = frozenset({'environment_check', 'unit_test', 'synthetic_smoke_test',
                   'real_training', 'real_validation', 'real_inference'})


@dataclass(frozen=True)
class TrainingConfig:
    artifact_kind: str = 'synthetic_smoke_test'
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    scheduler: str = 'linear'
    warmup_steps: int = 1
    gradient_clip_norm: float = 1.0
    seed: int = 1729
    device: str = 'cuda'
    precision: str = 'float32'
    deterministic_algorithms: bool = True
    shuffle: bool = True
    selection_metric: str = 'entity.micro.f1'
    selection_mode: str = 'max'
    min_delta: float = 0.0
    early_stopping_patience: int | None = None
    model_config_path: str = 'configs/bert_crf/stage1b_core_smoke.json'

    def __post_init__(self):
        for name in ('epochs', 'batch_size', 'seed', 'warmup_steps'):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in ('epochs', 'batch_size') else 0):
                raise ValueError(f'{name} must be an integer in range')
        for name in ('learning_rate', 'adam_epsilon', 'gradient_clip_norm', 'weight_decay', 'min_delta'):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
            if name in ('learning_rate', 'adam_epsilon', 'gradient_clip_norm') and value == 0:
                raise ValueError(f'{name} must be positive')
        if not isinstance(self.adam_betas, (tuple, list)) or len(self.adam_betas) != 2:
            raise ValueError('adam_betas must contain two numbers')
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < 1 for v in self.adam_betas):
            raise ValueError('adam_betas must be in [0,1)')
        object.__setattr__(self, 'adam_betas', tuple(self.adam_betas))
        if self.artifact_kind not in KINDS:
            raise ValueError('Unknown artifact_kind')
        if self.scheduler not in ('linear', 'constant') or self.selection_mode not in ('max', 'min'):
            raise ValueError('Unsupported scheduler/selection_mode')
        if self.selection_metric not in {'loss', *[f'{level}.{avg}.{measure}' for level in ('character', 'entity')
                for avg in ('micro', 'macro') for measure in ('precision', 'recall', 'f1')]}:
            raise ValueError('Unsupported selection_metric')
        if self.early_stopping_patience is not None and (type(self.early_stopping_patience) is not int or self.early_stopping_patience < 1):
            raise ValueError('patience must be null or positive integer')
        if type(self.shuffle) is not bool or type(self.deterministic_algorithms) is not bool:
            raise ValueError('shuffle/determinism must be bool')
        if self.precision != 'float32' or not isinstance(self.device, str) or self.device not in ('cpu', 'cuda'):
            raise ValueError('Only explicit CPU/CUDA float32 supported in this stage')
        if not isinstance(self.model_config_path, str) or not self.model_config_path:
            raise ValueError('model_config_path must be nonempty string')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))

    def to_dict(self):
        return asdict(self)


def require_synthetic(config, provider_kind=None):
    if config.artifact_kind != 'synthetic_smoke_test':
        raise RuntimeError('Real training/inference disabled: repaired public interface and real character Adapter are not integrated')
    if provider_kind is not None and provider_kind != config.artifact_kind:
        raise ValueError('Provider/config artifact_kind mismatch; no synthetic/real fallback')
