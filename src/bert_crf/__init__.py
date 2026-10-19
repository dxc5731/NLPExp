"""Character-level BERT-CRF core; real character alignment is not implemented."""
from .config import ModelConfig
from .constraints import LABELS, bio_masks
from .crf import LinearChainCRF

__all__ = ['ModelConfig', 'LABELS', 'bio_masks', 'LinearChainCRF']
