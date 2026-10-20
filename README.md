# NLPExp — IMCS-V2-NER 医疗命名实体识别

《自然语言处理》课程项目，方向 B（信息抽取 / 序列标注）。
数据：**IMCS-V2-NER**（CBLUE 医疗基准，公开数据集）。

任务：给定医患对话中的一句话，为每个字符打 BIO 标签，识别 5 类实体
（症状 / 药物 / 药物类别 / 检查 / 操作）。

## 分支说明

每个成员一个分支，各自维护，不合并到 `main`。

**本分支 `memberA` = 成员 A（刘天明）**，负责：

1. 公共数据处理：官方 train 按**整段对话**切分为内部 train/val（90:10，seed=42），
   名单在 `code/artifacts/split_v1.json`，全组共用。
2. BERT Tokenizer 与字符级 BIO 对齐（v2）：歧义词片标 -100 不伪造标签，
   全量 98529 句校验，明细在 `code/artifacts/alignment_report_*.json`。
3. 统一 `Dataset` / `DataLoader` 接口，供 BERT-Softmax 与 BERT-CRF 共用，
   说明见 `doc/组内有用的文档/BERT数据接口说明_v2.md`。
4. BERT-Softmax 模型（`code/src/model.py` / `code/src/train.py`）。

## 目录结构

```
code/
├─ src/          数据读取、分词对齐、Dataset、标签表、模型与训练
├─ tests/        自动测试（当前 13 个）
├─ artifacts/    全组共用的 train/val 名单、对齐检查报告
└─ requirements.txt
doc/             课程文档、组内分工与接口说明
handoff/         给组员的交接包（不入库，可重新生成）
```

## 当前进度

- [x] 数据体检（规模、标签分布、BIO 合法性、ID 唯一性）
- [x] 按**整段对话**切分 train/val，90:10，seed=42，名单见 `code/artifacts/split_v1.json`
- [x] BERT 子词与字符级 BIO 对齐 v2，全量 98529 句校验通过
- [x] 统一 Dataset / DataLoader 接口 + 13 个自动测试
- [ ] BERT-Softmax 训练（进行中）
- [ ] BERT-CRF（成员 B）
- [ ] dev 全量预测与交接

## 环境

Python 3.11 + PyTorch 2.6.0(cu124) + transformers 5.20.1，
详见 `code/requirements.txt`。

## 快速开始

1. 把 IMCS-V2-NER 数据集放到 `doc/数据/IMCS-V2-NER/IMCS-V2-NER/`，
   或者设环境变量 `IMCS_DATA_DIR` 指向它。
2. 装依赖：`cd code && pip install -r requirements.txt`
3. 验证接口：`python -m scripts.demo_dataloader`，看到 `一致: True` 即通。

## 约定

- 数据集与模型权重**不入库**（体积大，且课程要求数据集不进提交包）。
- 官方 **dev 不参与训练或调参**，只用于最终预测。
- 所有 ID 保持原样，不重新编号。
