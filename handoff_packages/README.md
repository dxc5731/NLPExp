# BERT-CRF 交接包说明

本目录是 BERT-CRF 模块（`src/bert_crf/`）的**结果交付包**。CRF 的预测和评分以 zip 形式保存，
仓库里没有平铺的预测文件，所以浏览代码时看不到——内容都在下面这两个包里。

| 文件 | 状态 |
|---|---|
| `BERT-CRF-LR1e-5-E5-Dev-Test-Handoff-v1.1_20261023T232759+0800.zip` | **当前版本，用这个** |
| `BERT-CRF-LR1e-5-E5-Dev-Test-Handoff_20261022T205207+0800.zip` | 早期版本，已被 v1.1 取代，仅作历史保留 |

## v1.1 里有什么

解压后是 `BERT-CRF-LR1e-5-E5-Dev-Test-Handoff-v1.1/`：

| 目录 | 内容 |
|---|---|
| `dev/` | `predictions_dev.json`、`run_info.json`、`validation_dev.json`、配置与日志 |
| `test/` | `predictions_test.json`、`run_info.json`、`validation_test.json`、配置与日志 |
| `reproduction_dev/` | 独立复现重跑，含 `unified_score/` 的官方评分结果 |
| `reproduce/` | 推理、校验、对比脚本与依赖锁文件 |
| `evidence/` | 环境快照、文件哈希、测试执行日志 |
| `model/` | checkpoint 引用与训练出处（权重本身不入库） |

## dev 上的结果（由 `evaluation/` 的评分器计算）

| 指标 | 值 |
|---|---|
| 实体级 P / R / F1 | 0.8757 / 0.9120 / **0.8935** |
| 字符准确率 | 0.9705 |
| 规模 | 833 对话 / 33267 句 / 441339 字符 |

评分结果原文：`reproduction_dev/unified_score/20261023T233622327709_iyp6tklb/evaluation.json`。

## 怎么用

```bash
# 解压出 dev 预测
unzip "BERT-CRF-LR1e-5-E5-Dev-Test-Handoff-v1.1_20261023T232759+0800.zip"

# 交给统一评估模块校验 / 评分
cd evaluation
python -X utf8 src/validate_predictions.py --data <数据> \
  --predictions <解压目录>/dev/predictions_dev.json --split dev --report <报告>
python -X utf8 src/evaluate_predictions.py --data <数据> \
  --predictions <解压目录>/dev/predictions_dev.json --split dev --purpose model_evaluation
```

## 说明

- `docs/bert_crf/` 下是**分阶段历史记录**，其中 2026-10-20 前后的文档写的"正式 Epoch 未启动"
  在当时是准确的；后续训练与交付以本目录的 v1.1 包和 `evaluation/` 的评分为准。
- 权重不入库，包内 `model/` 只给 checkpoint 的定位信息与哈希。
