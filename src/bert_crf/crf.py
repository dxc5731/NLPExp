"""Original batch-first linear-chain CRF using PyTorch dynamic programming.

No third-party CRF implementation is copied. Invalid arcs have exact -inf
scores in both the partition and Viterbi recurrences. Computation is fp32/fp64.
"""
import torch
from torch import nn

from .constraints import LABELS, bio_masks


class LinearChainCRF(nn.Module):
    def __init__(self, bio_constraints=True):
        super().__init__()
        self.num_tags = len(LABELS)
        self.bio_constraints = bio_constraints
        self.start_transitions = nn.Parameter(torch.empty(self.num_tags))
        self.end_transitions = nn.Parameter(torch.empty(self.num_tags))
        self.transitions = nn.Parameter(torch.empty(self.num_tags, self.num_tags))
        start, transition = bio_masks()
        if not bio_constraints:
            start = torch.ones_like(start)
            transition = torch.ones_like(transition)
        self.register_buffer('allowed_start', start, persistent=False)
        self.register_buffer('allowed_transition', transition, persistent=False)
        for p in self.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

    def _scores(self):
        return (self.start_transitions.masked_fill(~self.allowed_start, -torch.inf),
                self.transitions.masked_fill(~self.allowed_transition, -torch.inf))

    def _validate(self, emissions, mask, tags=None):
        if emissions.ndim != 3 or emissions.shape[-1] != self.num_tags:
            raise ValueError('emissions must have shape [B,L,11]')
        b, length, _ = emissions.shape
        if b == 0 or length == 0:
            raise ValueError('Empty batch/sequence is unsupported')
        if emissions.dtype not in (torch.float32, torch.float64):
            raise ValueError('CRF requires float32/float64 emissions')
        if mask.dtype != torch.bool or mask.shape != (b, length):
            raise ValueError('mask must be bool [B,L]')
        if mask.device != emissions.device or emissions.device != self.transitions.device:
            raise ValueError('CRF, mask and emissions must share device')
        if not bool(mask[:, 0].all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError('Each sample needs a nonempty continuous prefix mask')
        if not bool(torch.isfinite(emissions).all()):
            raise ValueError('Nonfinite emissions')
        if tags is not None:
            if tags.dtype != torch.long or tags.shape != mask.shape or tags.device != mask.device:
                raise ValueError('tags must be int64 [B,L] on same device')
            # Even padding must contain a valid placeholder (e.g. O), never -100.
            if bool(((tags < 0) | (tags >= self.num_tags)).any()):
                raise ValueError('Every tag including padding must be 0..10; -100 is forbidden')
            if not bool(self.allowed_start[tags[:, 0]].all()):
                raise ValueError('Illegal BIO gold start')
            valid = self.allowed_transition[tags[:, :-1], tags[:, 1:]]
            if bool((~valid & mask[:, 1:]).any()):
                raise ValueError('Illegal BIO gold transition')

    def _partition(self, emissions, mask):
        start, transition = self._scores()
        alpha = start + emissions[:, 0]
        for t in range(1, emissions.shape[1]):
            candidate = alpha.unsqueeze(2) + transition.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            next_alpha = torch.logsumexp(candidate, dim=1)
            alpha = torch.where(mask[:, t, None], next_alpha, alpha)
        return torch.logsumexp(alpha + self.end_transitions, dim=1)

    def _gold_score(self, emissions, tags, mask):
        start, transition = self._scores()
        batch = torch.arange(tags.shape[0], device=tags.device)
        score = start[tags[:, 0]] + emissions[batch, 0, tags[:, 0]]
        for t in range(1, tags.shape[1]):
            step = transition[tags[:, t - 1], tags[:, t]] + emissions[batch, t, tags[:, t]]
            # where, not multiplication: masked forbidden arcs may be -inf.
            score = score + torch.where(mask[:, t], step, torch.zeros_like(step))
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last]

    def log_partition(self, emissions, mask):
        self._validate(emissions, mask)
        return self._partition(emissions, mask)

    def sequence_score(self, emissions, tags, mask):
        self._validate(emissions, mask, tags)
        return self._gold_score(emissions, tags, mask)

    def forward(self, emissions, tags, mask, reduction='mean'):
        """Negative log likelihood; mean is over samples, not characters."""
        self._validate(emissions, mask, tags)
        nll = self._partition(emissions, mask) - self._gold_score(emissions, tags, mask)
        if reduction == 'none':
            return nll
        if reduction == 'sum':
            return nll.sum()
        if reduction == 'mean':
            return nll.mean()
        raise ValueError('reduction must be none/sum/mean')

    @torch.no_grad()
    def decode(self, emissions, mask):
        self._validate(emissions, mask)
        start, transition = self._scores()
        score = start + emissions[:, 0]
        history = []
        for t in range(1, emissions.shape[1]):
            best_score, best_previous = (score.unsqueeze(2) + transition.unsqueeze(0)).max(dim=1)
            score = torch.where(mask[:, t, None], best_score + emissions[:, t], score)
            history.append(best_previous)
        last = (score + self.end_transitions).argmax(dim=1)
        paths = []
        for b, length in enumerate(mask.sum(1).tolist()):
            tag = int(last[b])
            path = [tag]
            for t in range(length - 2, -1, -1):
                tag = int(history[t][b, tag])
                path.append(tag)
            paths.append(list(reversed(path)))
        return paths
