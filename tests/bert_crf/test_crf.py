import itertools
import json

import pytest
import torch

from bert_crf.constraints import LABELS
from bert_crf.crf import LinearChainCRF


def legal(path):
    """Independent path predicate, deliberately does not call production masks."""
    for t, tag in enumerate(path):
        name = LABELS[tag]
        if name.startswith('I-'):
            if t == 0:
                return False
            prev = LABELS[path[t - 1]]
            if prev == 'O' or prev[2:] != name[2:]:
                return False
    return True


def enumerated(crf, emissions, constrained):
    paths, scores = [], []
    for path in itertools.product(range(11), repeat=len(emissions)):
        if constrained and not legal(path):
            continue
        score = crf.start_transitions[path[0]] + crf.end_transitions[path[-1]]
        for t, tag in enumerate(path):
            score = score + emissions[t, tag]
            if t:
                score = score + crf.transitions[path[t - 1], tag]
        paths.append(list(path))
        scores.append(score)
    return paths, torch.stack(scores)


@pytest.mark.parametrize('constrained', [True, False])
@pytest.mark.parametrize('length', [1, 2, 3])
def test_exhaustive_partition_score_nll_viterbi_and_gradient(length, constrained):
    crf = LinearChainCRF(constrained).double()
    e = torch.randn(1, length, 11, dtype=torch.double, requires_grad=True)
    mask = torch.ones(1, length, dtype=torch.bool)
    paths, scores = enumerated(crf, e[0], constrained)
    expected_z = torch.logsumexp(scores, 0)
    gold = paths[len(paths) // 2]
    index = paths.index(gold)
    labels = torch.tensor([gold])
    actual_z = crf.log_partition(e, mask)[0]
    assert torch.allclose(actual_z, expected_z, atol=1e-12, rtol=1e-12)
    assert torch.allclose(crf.sequence_score(e, labels, mask)[0], scores[index], atol=1e-12, rtol=1e-12)
    nll = crf(e, labels, mask)
    assert torch.allclose(nll, expected_z - scores[index], atol=1e-12, rtol=1e-12)
    assert crf.decode(e, mask) == [paths[int(scores.argmax())]]
    print(json.dumps({'check': 'exhaustive_synthetic_paths', 'length': length,
                      'bio_constraints': constrained, 'included_paths': len(paths),
                      'partition_dp': float(actual_z.detach()),
                      'partition_enumeration': float(expected_z.detach()),
                      'gold_path': gold, 'gold_score': float(scores[index].detach()),
                      'nll': float(nll.detach()), 'best_path': crf.decode(e, mask)[0]}))
    parameters = (e, crf.start_transitions, crf.end_transitions, crf.transitions)
    g_actual = torch.autograd.grad(nll, parameters, retain_graph=True, allow_unused=True)
    g_expected = torch.autograd.grad(expected_z - scores[index], parameters, allow_unused=True)
    for actual, expected in zip(g_actual, g_expected):
        if actual is None:
            assert expected is None  # Length 1 has no transitions.
        else:
            assert torch.isfinite(actual).all()
            assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_variable_batch_padding_invariant_loss_path_and_gradients():
    crf = LinearChainCRF().double()
    e = torch.randn(3, 4, 11, dtype=torch.double, requires_grad=True)
    lengths = [4, 2, 1]
    mask = torch.arange(4)[None, :] < torch.tensor(lengths)[:, None]
    tags = torch.tensor([[1, 2, 0, 3], [5, 6, 2, 4], [0, 10, 8, 2]])
    batched = crf(e, tags, mask, reduction='none')
    paths = crf.decode(e, mask)
    assert [len(p) for p in paths] == lengths
    assert all(legal(p) for p in paths)
    for b, length in enumerate(lengths):
        assert torch.allclose(batched[b], crf(e[b:b+1, :length], tags[b:b+1, :length], mask[b:b+1, :length]))
    changed = e.detach().clone()
    changed[~mask] = 1e6
    assert torch.equal(batched, crf(changed, tags, mask, reduction='none'))
    assert paths == crf.decode(changed, mask)
    assert torch.allclose(batched.mean(), crf(e, tags, mask))
    assert torch.allclose(batched.sum(), crf(e, tags, mask, reduction='sum'))
    batched.sum().backward()
    assert torch.isfinite(e.grad).all() and bool((e.grad[~mask] == 0).all())


@pytest.mark.parametrize('case', ['empty_batch', 'empty_time', 'zero_length_sample', 'first_false', 'hole',
                                 'mask_int', 'mask_shape', 'emission_shape', 'half', 'nan', 'inf',
                                 'tag_float', 'tag_shape', 'negative', 'too_large', 'padding_minus100'])
def test_reject_bad_inputs(case):
    crf = LinearChainCRF()
    e = torch.zeros(2, 3, 11)
    mask = torch.ones(2, 3, dtype=torch.bool)
    tags = torch.zeros(2, 3, dtype=torch.long)
    if case == 'empty_batch': e, mask, tags = e[:0], mask[:0], tags[:0]
    if case == 'empty_time': e, mask, tags = e[:, :0], mask[:, :0], tags[:, :0]
    if case == 'zero_length_sample': mask[1] = False
    if case == 'first_false': mask[:, 0] = False
    if case == 'hole': mask[:, 1] = False
    if case == 'mask_int': mask = mask.long()
    if case == 'mask_shape': mask = mask[:, :2]
    if case == 'emission_shape': e = e[..., :10]
    if case == 'half': e = e.half()
    if case == 'nan': e[0, 0, 0] = float('nan')
    if case == 'inf': e[0, 0, 0] = float('inf')
    if case == 'tag_float': tags = tags.float()
    if case == 'tag_shape': tags = tags[:, :2]
    if case == 'negative': tags[0, 0] = -100
    if case == 'too_large': tags[0, 0] = 11
    if case == 'padding_minus100': mask[0, -1], tags[0, -1] = False, -100
    with pytest.raises(ValueError):
        crf(e, tags, mask)


def test_invalid_reduction():
    with pytest.raises(ValueError, match='reduction'):
        LinearChainCRF()(torch.zeros(1, 1, 11), torch.zeros(1, 1, dtype=torch.long),
                         torch.ones(1, 1, dtype=torch.bool), reduction='unknown')


def test_extreme_finite_emissions_are_stable():
    e = (torch.randn(2, 20, 11) * 1e4).requires_grad_()
    crf = LinearChainCRF()
    loss = crf(e, torch.zeros(2, 20, dtype=torch.long), torch.ones(2, 20, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(e.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in crf.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_gpu_cpu_crf_agreement_and_backward():
    cpu = LinearChainCRF()
    gpu = LinearChainCRF().cuda()
    gpu.load_state_dict(cpu.state_dict())
    e = torch.randn(2, 5, 11, requires_grad=True)
    eg = e.detach().cuda().requires_grad_()
    tags = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 0, 0, 0]])
    mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    lc = cpu(e, tags, mask)
    lg = gpu(eg, tags.cuda(), mask.cuda())
    lc.backward()
    lg.backward()
    assert torch.allclose(lc, lg.cpu(), atol=1e-5)
    assert torch.allclose(e.grad, eg.grad.cpu(), atol=1e-6)
    assert cpu.decode(e, mask) == gpu.decode(eg, mask.cuda())
    assert all(torch.isfinite(p.grad).all() for p in gpu.parameters())
    with pytest.raises(ValueError, match='share device'):
        gpu(eg, tags, mask)
