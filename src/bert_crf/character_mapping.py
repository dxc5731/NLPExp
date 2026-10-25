"""Stateless, differentiable offsets sharing. No labels, repairs or fallbacks."""
import torch

from .model import CharacterRepresentationAdapter

OFFSET_UNIT = 'python_unicode_codepoint'
MAPPING_POLICY = 'offset_share_unique_v1'


class MappingError(ValueError):
    """Structured, text-free rejection, safe for diagnostic logs."""
    def __init__(self, reason, *, sample=None, positions=()):
        self.reason, self.sample, self.positions = reason, sample, list(positions)
        super().__init__(f'{reason}: sample={sample}, positions={self.positions[:16]}')


def character_index(offsets, attention_mask, character_lengths, special_tokens_mask, *,
                    offset_unit, policy=MAPPING_POLICY):
    """Validate Python code-point intervals; return [B,C] token indices and mask.

    The caller must verify coordinate provenance using a Fast Tokenizer against
    the original strings (V2DataAdapter does so). A declaration alone does not
    establish provenance. Only inserted special tokens and PAD may be zero-width;
    literal special-token strings with nonzero offsets remain original characters.
    """
    if offset_unit != OFFSET_UNIT or policy != MAPPING_POLICY:
        raise MappingError('unconfirmed_offset_unit_or_policy')
    if not isinstance(offsets, torch.Tensor) or offsets.dtype != torch.long or offsets.ndim != 3 or offsets.shape[-1] != 2:
        raise MappingError('offset_shape_or_dtype')
    b, t, _ = offsets.shape
    if b == 0 or t == 0:
        raise MappingError('empty_batch_or_tokens')
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != (b, t) or attention_mask.dtype not in (torch.bool, torch.long):
        raise MappingError('attention_shape_or_dtype')
    if not isinstance(special_tokens_mask, torch.Tensor) or special_tokens_mask.shape != (b, t) or special_tokens_mask.dtype != torch.bool:
        raise MappingError('special_mask_shape_or_dtype')
    if not isinstance(character_lengths, torch.Tensor) or character_lengths.shape != (b,) or character_lengths.dtype != torch.long:
        raise MappingError('length_shape_or_dtype')
    if len({x.device for x in (offsets, attention_mask, character_lengths, special_tokens_mask)}) != 1:
        raise MappingError('mapping_device_mismatch')
    lengths = character_lengths.detach().cpu().tolist()
    if any(n <= 0 for n in lengths):
        raise MappingError('empty_character_sequence')
    attention = attention_mask.detach().cpu()
    if bool(((attention != 0) & (attention != 1)).any()):
        raise MappingError('nonbinary_attention')
    attention = attention.bool()
    if not bool(attention[:, 0].all()) or bool((attention[:, 1:] & ~attention[:, :-1]).any()):
        raise MappingError('nonprefix_attention')
    spans = offsets.detach().cpu().tolist()
    specials = special_tokens_mask.detach().cpu().tolist()
    # Validate bounds before allocation to avoid trusting an unbounded length.
    for row, n in enumerate(lengths):
        for j, (start, end) in enumerate(spans[row]):
            if start < 0 or end < start or end > n:
                raise MappingError('invalid_offset', sample=row, positions=[j])
            if not attention[row, j]:
                if (start, end) != (0, 0):
                    raise MappingError('padding_has_span', sample=row, positions=[j])
            elif specials[row][j]:
                if (start, end) != (0, 0):
                    raise MappingError('special_has_span', sample=row, positions=[j])
            elif start == end:
                raise MappingError('content_has_empty_span', sample=row, positions=[j])
    indices = torch.zeros((b, max(lengths)), dtype=torch.long)
    mask = torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    for row, n in enumerate(lengths):
        coverage = [0] * n
        previous_start = -1
        for j, (start, end) in enumerate(spans[row]):
            if not attention[row, j] or specials[row][j]:
                continue
            if start < previous_start:
                raise MappingError('offset_order', sample=row, positions=[j])
            previous_start = start
            for c in range(start, end):
                coverage[c] += 1
                indices[row, c] = j
        overlap = [c for c, k in enumerate(coverage) if k > 1]
        gaps = [c for c, k in enumerate(coverage) if k == 0]
        if overlap:
            raise MappingError('coverage_overlap', sample=row, positions=overlap)
        if gaps:
            raise MappingError('coverage_gap', sample=row, positions=gaps)
    return indices.to(offsets.device), mask.to(offsets.device)


class OffsetCharacterAdapter(CharacterRepresentationAdapter):
    """Each character gathers its unique source token; shared gradients add up.

    No parameters/buffers, compatible with existing model checkpoints. Persist
    MAPPING_POLICY/offset unit and source hashes in provider/run metadata; loading
    a model still requires explicitly supplying this same adapter implementation.
    """
    policy = MAPPING_POLICY

    def forward(self, token_hidden_states, *, character_mask, offsets, attention_mask,
                character_lengths, special_tokens_mask, offset_unit, policy=MAPPING_POLICY):
        if not isinstance(token_hidden_states, torch.Tensor) or token_hidden_states.ndim != 3 or not token_hidden_states.is_floating_point():
            raise MappingError('hidden_shape_or_dtype')
        if not isinstance(offsets, torch.Tensor) or token_hidden_states.shape[:2] != offsets.shape[:2]:
            raise MappingError('hidden_token_shape_mismatch')
        indices, expected = character_index(offsets, attention_mask, character_lengths,
            special_tokens_mask, offset_unit=offset_unit, policy=policy)
        if not isinstance(character_mask, torch.Tensor) or character_mask.dtype != torch.bool or character_mask.shape != expected.shape:
            raise MappingError('character_mask_shape_or_dtype')
        if token_hidden_states.device != offsets.device or character_mask.device != offsets.device:
            raise MappingError('hidden_or_mask_device_mismatch')
        if not torch.equal(character_mask, expected):
            raise MappingError('character_mask_length_mismatch')
        characters = token_hidden_states.gather(1, indices.unsqueeze(-1).expand(-1, -1, token_hidden_states.shape[-1]))
        return characters.masked_fill(~character_mask.unsqueeze(-1), 0)
