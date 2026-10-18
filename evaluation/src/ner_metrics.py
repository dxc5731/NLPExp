"""Pure strict BIO span and Unicode-position classification metrics (stdlib).

All numbers are fractions in [0, 1]. Every zero denominator yields 0.0.
No text normalization, BIO repair, file I/O, or model inference occurs here.
"""

from dataclasses import dataclass

if __package__:
    from .validate_predictions import ENTITY_TYPES, LABELS
else:
    from validate_predictions import ENTITY_TYPES, LABELS


LABEL_ORDER = ("O",) + tuple(f"{prefix}-{kind}" for kind in ENTITY_TYPES for prefix in ("B", "I"))
LABEL_INDEX = {tag: index for index, tag in enumerate(LABEL_ORDER)}


@dataclass(frozen=True)
class Entity:
    dialogue_id: str
    sentence_id: str
    entity_type: str
    start: int
    end: int  # exclusive, original Unicode code-point coordinates


def prf_counts(tp, fp, fn):
    """Return unrounded metrics and sufficient counts; undefined ratios are 0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "support": tp + fn, "predicted": tp + fp}


def extract_entities(tags, dialogue_id, sentence_id):
    """Extract strict BIO entities with [start, end); reject illegal sequences."""
    if isinstance(tags, (str, bytes)):
        raise ValueError("tags must be a sequence of individual BIO labels")
    if not isinstance(dialogue_id, str) or not dialogue_id or not isinstance(sentence_id, str) or not sentence_id:
        raise ValueError("original dialogue_id and sentence_id must be nonempty strings")
    entities = set()
    start = None
    kind = None
    previous = "O"
    length = 0
    for index, tag in enumerate(tags):
        length = index + 1
        if not isinstance(tag, str) or tag not in LABELS:
            raise ValueError(f"invalid BIO label at position {index}")
        if tag.startswith("I-") and previous not in ("B-" + tag[2:], tag):
            raise ValueError(f"illegal I transition at position {index}")
        if tag == "O" or tag.startswith("B-"):
            if start is not None:
                entities.add(Entity(dialogue_id, sentence_id, kind, start, index))
                start = kind = None
            if tag.startswith("B-"):
                start, kind = index, tag[2:]
        previous = tag
    if start is not None:
        entities.add(Entity(dialogue_id, sentence_id, kind, start, length))
    return entities


def score_entity_sets(gold, predicted):
    """Exact type, boundary, dialogue and sentence match, pooled over all spans."""
    gold, predicted = set(gold), set(predicted)
    for entity in gold | predicted:
        if not isinstance(entity, Entity) or entity.entity_type not in ENTITY_TYPES:
            raise ValueError("unknown entity type or non-Entity input")
    right = gold & predicted
    per_type = {}
    for kind in ENTITY_TYPES:
        g = sum(e.entity_type == kind for e in gold)
        p = sum(e.entity_type == kind for e in predicted)
        tp = sum(e.entity_type == kind for e in right)
        per_type[kind] = prf_counts(tp, p - tp, g - tp)
    tp = len(right)
    return {"micro": prf_counts(tp, len(predicted) - tp, len(gold) - tp),
            "per_type": per_type, "gold_total": len(gold), "predicted_total": len(predicted)}


def score_confusion_matrix(matrix):
    """Rows=gold, columns=predicted, fixed LABEL_ORDER, no O-position filtering."""
    size = len(LABEL_ORDER)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("confusion matrix must be 11 by 11")
    if any(type(n) is not int or n < 0 for row in matrix for n in row):
        raise ValueError("confusion counts must be nonnegative integers")
    matrix = [list(row) for row in matrix]
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(row[j] for row in matrix) for j in range(size)]
    per_label = {tag: prf_counts(matrix[i][i], col_sums[i] - matrix[i][i],
                                 row_sums[i] - matrix[i][i])
                 for i, tag in enumerate(LABEL_ORDER)}

    def summary(labels):
        metrics = [per_label[tag] for tag in labels]
        micro = prf_counts(*(sum(m[key] for m in metrics) for key in ("tp", "fp", "fn")))
        return {"labels": list(labels), "micro": micro,
                "macro_f1": sum(m["f1"] for m in metrics) / len(labels),
                "macro_denominator": len(labels)}

    total = sum(row_sums)
    correct = sum(matrix[i][i] for i in range(size))
    return {"total": total, "correct": correct, "accuracy": correct / total if total else 0.0,
            "per_label": per_label, "including_o": summary(LABEL_ORDER),
            "excluding_o": summary(LABEL_ORDER[1:]),
            "confusion_matrix": {"labels": list(LABEL_ORDER), "rows": "gold", "columns": "predicted",
                                 "counts": matrix, "row_sums": row_sums, "column_sums": col_sums}}


def score_sequences(gold, predicted):
    """Score {(dialogue_id, sentence_id): sequence_of_tags} mappings.

    Pure API also rejects ID/length/BIO errors; file consumers must additionally
    pass the stage-one full data/prediction validation before calling this API.
    """
    if gold.keys() != predicted.keys():
        raise ValueError("gold and predicted sentence IDs must match exactly")
    matrix = [[0] * len(LABEL_ORDER) for _ in LABEL_ORDER]
    gold_entities, predicted_entities = set(), set()
    for key, gold_tags in gold.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("sentence key must be (dialogue_id, sentence_id)")
        pred_tags = predicted[key]
        if len(gold_tags) != len(pred_tags):
            raise ValueError("gold and predicted sequence lengths differ")
        gold_entities.update(extract_entities(gold_tags, *key))
        predicted_entities.update(extract_entities(pred_tags, *key))
        for g, p in zip(gold_tags, pred_tags):
            matrix[LABEL_INDEX[g]][LABEL_INDEX[p]] += 1
    return {"entities": score_entity_sets(gold_entities, predicted_entities),
            "characters": score_confusion_matrix(matrix)}
