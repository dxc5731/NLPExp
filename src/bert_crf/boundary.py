"""Class-agnostic auxiliary boundaries; CRF decoding is never modified."""
import torch
from torch import nn
from torch.nn import functional as F

from .metrics import entities, prf


def boundary_targets(labels, mask):
    if labels.dtype != torch.long or mask.dtype != torch.bool or labels.shape != mask.shape or mask.ndim != 2:
        raise ValueError('Boundary targets require int64 labels and bool character mask [B,C]')
    if 0 in mask.shape or not bool(mask[:, 0].all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError('Boundary targets require nonempty prefix character masks')
    starts = torch.zeros_like(labels, dtype=torch.float32)
    ends = torch.zeros_like(starts)
    rows, lengths = labels.detach().cpu().tolist(), mask.sum(1).tolist()
    for b, length in enumerate(lengths):
        for start, end, _ in entities(rows[b][:length]):
            starts[b, start], ends[b, end - 1] = 1., 1.
    return starts, ends


def balanced_bce(logits, targets, mask):
    if logits.shape != mask.shape or targets.shape != mask.shape or not bool(mask.any()):
        raise ValueError('Boundary loss shape/mask mismatch')
    # Select before BCE: even arbitrary padding logits/targets cannot enter loss.
    values, gold = logits[mask], targets[mask]
    if not bool(torch.isfinite(values).all()) or not bool(((gold == 0) | (gold == 1)).all()):
        raise ValueError('Invalid valid-character boundary logits/targets')
    losses = F.binary_cross_entropy_with_logits(values, gold, reduction='none')
    positive = gold == 1
    parts = [losses[part].mean() for part in (positive, ~positive) if bool(part.any())]
    return torch.stack(parts).mean()


class BoundaryAuxHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.start = nn.Linear(hidden_size, 1)
        self.end = nn.Linear(hidden_size, 1)

    def forward(self, characters):
        return self.start(characters).squeeze(-1), self.end(characters).squeeze(-1)


def boundary_counts(output, labels, mask):
    targets = boundary_targets(labels, mask)
    result = {}
    for name, target in zip(('start', 'end'), targets):
        predicted = output[name + '_logits'][mask] >= 0  # fixed sigmoid threshold0.5
        gold = target[mask].bool()
        result[name] = {'tp': int((predicted & gold).sum()), 'predicted': int(predicted.sum()), 'gold': int(gold.sum())}
    return result


def boundary_metrics(counts):
    return {name: prf(**values) for name, values in counts.items()}
