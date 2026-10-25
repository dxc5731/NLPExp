import pytest
import torch

from bert_crf.constraints import LABELS, bio_masks
from bert_crf.crf import LinearChainCRF


def test_all_pairs_against_independent_bio_rule():
    start, transitions = bio_masks()
    assert start.tolist() == [True, True, False, True, False, True, False, True, False, True, False]
    assert int(transitions.sum()) == 76
    for i, previous in enumerate(LABELS):
        for j, current in enumerate(LABELS):
            expected = current[0] != 'I' or (previous[0] in 'BI' and previous[2:] == current[2:])
            assert bool(transitions[i, j]) == expected


@pytest.mark.parametrize('tags', [[1, 2], [1, 1], [0, 0], [3], [1, 3, 4], [5, 6, 7, 8, 9, 10]])
def test_legal_gold(tags):
    crf = LinearChainCRF()
    labels = torch.tensor([tags])
    assert torch.isfinite(crf(torch.zeros(1, len(tags), 11), labels, torch.ones_like(labels, dtype=torch.bool)))


@pytest.mark.parametrize('tags', [[2], [0, 2], [1, 4], [3, 2], [1, 2, 6]])
def test_illegal_gold_refused(tags):
    crf = LinearChainCRF()
    labels = torch.tensor([tags])
    with pytest.raises(ValueError, match='Illegal BIO gold'):
        crf(torch.zeros(1, len(tags), 11), labels, torch.ones_like(labels, dtype=torch.bool))


def test_illegal_arcs_cannot_win_or_enter_partition():
    crf = LinearChainCRF().double()
    mask = torch.ones(1, 2, dtype=torch.bool)
    emissions = torch.zeros(1, 2, 11, dtype=torch.double)
    baseline = crf.log_partition(emissions, mask).detach().clone()
    with torch.no_grad():
        crf.start_transitions[2] = 1e6
        crf.transitions[0, 2] = 1e6
        crf.transitions[3, 2] = 1e6
    assert torch.equal(baseline, crf.log_partition(emissions, mask))
    path = crf.decode(emissions, mask)[0]
    assert path[0] != 2 and path != [0, 2] and path != [3, 2]
    crf.log_partition(emissions, mask).sum().backward()
    assert crf.start_transitions.grad[2] == 0
    assert crf.transitions.grad[0, 2] == 0
    assert crf.transitions.grad[3, 2] == 0
