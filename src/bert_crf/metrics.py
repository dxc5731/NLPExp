"""Internal character/BIO and strict entity metrics. Not the group's final evaluator."""
from collections import Counter

from .constraints import LABELS

TYPES = tuple(tag[2:] for tag in LABELS if tag.startswith('B-'))


def checked_tags(tags):
    result = []
    for value in tags:
        if type(value) is int and 0 <= value < len(LABELS):
            tag = LABELS[value]
        elif isinstance(value, str) and value in LABELS:
            tag = value
        else:
            raise ValueError(f'Unknown BIO label: {value!r}')
        if tag.startswith('I-') and (not result or result[-1] not in ('B-' + tag[2:], 'I-' + tag[2:])):
            raise ValueError('Illegal BIO sequence; metrics never repair labels')
        result.append(tag)
    if not result:
        raise ValueError('Empty character sequence')
    return result


def entities(tags):
    """Half-open [start,end) spans in original Python character positions."""
    tags = checked_tags(tags)
    result = set()
    start, category = None, None
    for i, tag in enumerate([*tags, 'O']):
        if start is not None and not tag.startswith('I-'):
            result.add((start, i, category))
            start, category = None, None
        if tag.startswith('B-'):
            start, category = i, tag[2:]
    return result


def prf(tp, predicted, gold):
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    return {'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'true_positive': tp, 'predicted': predicted, 'gold': gold}


class Metrics:
    """Streaming sentence-scoped counting; duplicate (dialogue_id,sentence_id) rejects.

    Character TP requires exact BIO-tag equality. O excluded from P/R/F1;
    per-type counts group B/I without forgiving B-vs-I mismatches. Macro means
    unweighted mean over all five fixed types, including unsupported types (0).
    Accuracy includes O and is reported separately. All zero denominators -> 0.
    """
    def __init__(self):
        self.seen = set()
        self.counts = {level: {kind: Counter() for kind in ('tp', 'predicted', 'gold')}
                       for level in ('character', 'entity')}
        self.characters = self.correct = 0

    def update(self, sentence_key, gold, predicted):
        if not isinstance(sentence_key, tuple) or len(sentence_key) != 2 or any(not isinstance(x, str) or not x for x in sentence_key):
            raise ValueError('Sentence key must be (nonempty original dialogue_id, sentence_id) strings')
        if sentence_key in self.seen:
            raise ValueError('Duplicate sentence ID in evaluation')
        gold, predicted = checked_tags(gold), checked_tags(predicted)
        if len(gold) != len(predicted):
            raise ValueError('Gold/prediction character lengths differ')
        self.seen.add(sentence_key)
        self.characters += len(gold)
        self.correct += sum(g == p for g, p in zip(gold, predicted))
        c = self.counts['character']
        for g, p in zip(gold, predicted):
            if g != 'O': c['gold'][g[2:]] += 1
            if p != 'O': c['predicted'][p[2:]] += 1
            if g == p and g != 'O': c['tp'][g[2:]] += 1
        ge, pe = entities(gold), entities(predicted)
        c = self.counts['entity']
        for kind, items in [('gold', ge), ('predicted', pe), ('tp', ge & pe)]:
            c[kind].update(item[2] for item in items)

    def compute(self):
        result = {'sentences': len(self.seen), 'characters': self.characters,
                  'character_accuracy_including_O': self.correct / self.characters if self.characters else 0.0,
                  'protocol': 'strict_bio_metrics_v1; exact BIO characters excluding O; fixed-five-type macro; zero division=0'}
        for level, c in self.counts.items():
            per_type = {t: prf(c['tp'][t], c['predicted'][t], c['gold'][t]) for t in TYPES}
            result[level] = {'micro': prf(sum(c['tp'].values()), sum(c['predicted'].values()), sum(c['gold'].values())),
                             'macro': {m: sum(x[m] for x in per_type.values()) / len(TYPES) for m in ('precision', 'recall', 'f1')},
                             'per_type': per_type}
        return result
