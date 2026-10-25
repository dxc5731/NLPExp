# -*- coding: utf-8 -*-
"""标签表：全项目只在这一处定义，别的地方一律 import。

顺序（下标）一旦定下就不要改，因为模型输出层的第 i 个神经元就对应这里第 i 个标签。
"""

LABELS = [
    "O",
    "B-Symptom", "I-Symptom",
    "B-Drug", "I-Drug",
    "B-Drug_Category", "I-Drug_Category",
    "B-Medical_Examination", "I-Medical_Examination",
    "B-Operation", "I-Operation",
]

LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for i, name in enumerate(LABELS)}

ENTITY_TYPES = ["Symptom", "Drug", "Drug_Category", "Medical_Examination", "Operation"]
