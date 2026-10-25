import pytest

from bert_crf.metrics import Metrics, entities


def score(gold, predicted):
    metrics = Metrics()
    metrics.update(('dialogue', '01'), gold, predicted)
    return metrics.compute()


def test_known_character_and_strict_entity_counts():
    # Gold: Symptom [0,2), Drug [3,4). Prediction truncates Symptom to [0,1).
    result = score([1, 2, 0, 3], [1, 0, 0, 3])
    assert result['character']['micro'] == {'precision': 1.0, 'recall': 2/3, 'f1': 0.8,
        'true_positive': 2, 'predicted': 2, 'gold': 3}
    assert result['entity']['micro'] == {'precision': 0.5, 'recall': 0.5, 'f1': 0.5,
        'true_positive': 1, 'predicted': 2, 'gold': 2}
    assert result['character']['macro']['f1'] == pytest.approx((2/3 + 1) / 5)
    assert result['entity']['macro']['f1'] == 0.2
    assert result['character_accuracy_including_O'] == 0.75


def test_adjacent_same_type_is_two_entities_and_bio_mismatch_is_character_error():
    assert entities([1, 1]) == {(0, 1, 'Symptom'), (1, 2, 'Symptom')}
    r = score([1, 1], [1, 2])
    assert r['character']['micro']['f1'] == 0.5
    assert r['entity']['micro']['f1'] == 0


def test_wrong_type_no_partial_credit():
    r = score([1, 2], [3, 4])
    assert r['entity']['micro']['true_positive'] == 0
    assert r['character']['micro']['f1'] == 0
    assert r['entity']['per_type']['Drug']['predicted'] == 1


@pytest.mark.parametrize('gold,predicted', [([0, 0], [0, 0]), ([1, 2], [0, 0]), ([0, 0], [1, 2])])
def test_zero_denominators(gold, predicted):
    r = score(gold, predicted)
    for level in ('entity', 'character'):
        assert all(r[level]['micro'][key] == 0 for key in ('precision', 'recall', 'f1'))


def test_single_character_all_categories_micro_macro():
    r = score([1, 3, 5, 7, 9], [1, 3, 5, 7, 9])
    for level in ('character', 'entity'):
        assert r[level]['micro']['f1'] == r[level]['macro']['f1'] == 1


def test_sentence_and_dialogue_ids_prevent_cross_sentence_matches():
    m = Metrics()
    m.update(('d1', '01'), [1], [0])
    m.update(('d2', '01'), [0], [1])
    m.update(('d2', '1'), [0], [0])
    assert m.compute()['entity']['micro']['f1'] == 0
    assert m.compute()['sentences'] == 3
    with pytest.raises(ValueError, match='Duplicate'):
        m.update(('d1', '01'), [1], [1])


@pytest.mark.parametrize('tags', [[2], [0, 2], [1, 4], [-100], [11], ['I-Drug'], ['bad'], [True], []])
def test_invalid_bio_refused(tags):
    with pytest.raises(ValueError):
        entities(tags)


def test_metric_length_and_id_validation():
    with pytest.raises(ValueError, match='lengths'):
        score([0, 0], [0])
    with pytest.raises(ValueError, match='Sentence key'):
        Metrics().update(('d', 1), [0], [0])


def test_accumulation_equivalence():
    m = Metrics()
    m.update(('d', '1'), [1, 2], [1, 2])
    m.update(('d', '2'), [3], [0])
    r = m.compute()
    assert r['entity']['micro']['precision'] == 1
    assert r['entity']['micro']['recall'] == 0.5
    assert r['character']['micro']['recall'] == 2/3
