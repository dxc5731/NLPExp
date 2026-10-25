"""Strict BIO v1; transition tensors are indexed [previous_tag, next_tag]."""
import torch

LABELS = ('O', 'B-Symptom', 'I-Symptom', 'B-Drug', 'I-Drug',
          'B-Drug_Category', 'I-Drug_Category', 'B-Medical_Examination',
          'I-Medical_Examination', 'B-Operation', 'I-Operation')


def bio_masks():
    start = torch.tensor([not tag.startswith('I-') for tag in LABELS], dtype=torch.bool)
    transition = torch.tensor([
        [not nxt.startswith('I-') or prev in ('B-' + nxt[2:], 'I-' + nxt[2:])
         for nxt in LABELS] for prev in LABELS
    ], dtype=torch.bool)
    return start, transition
