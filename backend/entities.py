"""Reuse the project's strict scorer; no alternative formal scoring policy."""
from pathlib import Path
import sys
from .common import ROOT

sys.path.insert(0, str(ROOT / 'evaluation/src'))
from ner_metrics import extract_entities, score_entity_sets

def entities(text, tags):
    if len(text) != len(tags): raise ValueError('字符与标签长度不一致')
    return [{'text': text[e.start:e.end], 'entity_type': e.entity_type, 'start': e.start,
             'end': e.end, 'tags': tags[e.start:e.end]}
            for e in sorted(extract_entities(tags, 'local', 'input'), key=lambda e: e.start)]

def strict_score(gold_tags, pred_tags):
    return score_entity_sets(extract_entities(gold_tags, 'local', 'input'),
                             extract_entities(pred_tags, 'local', 'input'))['micro']

def differences(results):
    ok = [r for r in results if r['status'] == 'success']
    groups = {}
    for r in ok:
        for e in r['entities']:
            key = (e['start'], e['end'], e['entity_type'])
            groups.setdefault(key, {**e, 'methods': []})['methods'].append(r['method'])
    values = list(groups.values())
    boundary, types = [], []
    for i, a in enumerate(values):
        for b in values[i+1:]:
            if not any(x != y for x in a['methods'] for y in b['methods']): continue
            if (a['start'], a['end']) == (b['start'], b['end']) and a['entity_type'] != b['entity_type']:
                types.append([a, b])
            elif max(a['start'], b['start']) < min(a['end'], b['end']) and a['entity_type'] == b['entity_type']:
                boundary.append([a, b])
    return {'shared_all': [e for e in values if len(e['methods']) == 3],
            'method_only': [e for e in values if len(e['methods']) == 1],
            'overlapping_boundary_differences': boundary, 'same_boundary_type_differences': types,
            'counts': {r['method']: len(r['entities']) for r in ok},
            'compared_methods': [r['method'] for r in ok],
            'notice': '按原始字符[start,end)与类型比较；重叠边界项是几何差异，不推断语义原因。'}
