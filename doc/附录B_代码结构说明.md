# 附录 B：代码结构说明

> 仓库由三条独立开发的分支合并而成，按模块分目录。顶层共 488 个文件。

## 顶层目录

| 目录 / 文件 | 内容 |
|---|---|
| `code/` | **公共数据 + BERT-Softmax**：切分、子词对齐、Dataset、训练、预测、导出 |
| `src/` `configs/` `docs/` `tests/` | **BERT-CRF**：模型核心、配置、阶段文档与测试 |
| `evaluation/` | **统一评估**：预测校验、评分、错误导出、复现证据核查、多模型比较 |
| `doc/` | 课程文档、组内分工与接口说明、实验记录、耗时参考、附录材料 |
| `deploy/` | 服务器训练包脚本（setup / run / collect） |
| `runs/` | 各次实验的配置、日志、指标（权重不入库） |
| `deploy/` `handoff_packages/` | 交接包与生成工具 |

## `code/`：公共数据与 BERT-Softmax

```
code/
├─ src/
│  ├─ config.py      路径与常量；数据/模型目录可用环境变量覆盖；缺目录时给明确提示
│  ├─ labels.py      11 种 BIO 标签，全局唯一定义
│  ├─ data_io.py     读原始 JSON、算文件哈希
│  ├─ split.py       按整段对话切分 train/val（seed=42，90:10）
│  ├─ align.py       字符级 BIO ↔ BERT 词片对齐；歧义词片标 -100，不伪造标签
│  ├─ dataset.py     Dataset / DataLoader / collate，供 Softmax 与 CRF 共用
│  ├─ model.py       bert-base-chinese + 11 类线性分类头
│  ├─ metrics.py     逐 token 指标 + 实体级指标（seqeval）
│  ├─ train.py       训练循环、按实体 F1 存最优、写日志与配置
│  └─ predict.py     推理导出 predictions_*.json 与 run_info.json
├─ scripts/          一次性工具（数据体检、对齐校验、自检、打包、天池格式转换）
├─ tests/            13 个单元测试（对齐、BIO 合法性、padding、已知歧义句回归）
├─ artifacts/        全组共用的 split_v1.json、三份对齐报告
└─ requirements*.txt 依赖清单
```

## `src/` `configs/` `docs/` `tests/`：BERT-CRF

| 路径 | 内容 |
|---|---|
| `src/bert_crf/` | 编码器、字符级 Dropout/Linear/CRF、BIO 硬约束、训练器 |
| `configs/bert_crf/` | 基础配置、学习率配置、阶段配置 |
| `docs/bert_crf/` | 阶段 0—1E 的审计、实现、验收与证据记录 |
| `tests/bert_crf/` | 单元与集成测试 |

详细说明见 [`docs/bert_crf/README.md`](../docs/bert_crf/README.md)。

## `evaluation/`：统一评估

| 路径 | 内容 |
|---|---|
| `evaluation/src/validate_predictions.py` | 预测格式、ID 集合、逐字符长度、BIO 合法性校验 |
| `evaluation/src/evaluate_predictions.py` | 实体级（strict）与字符级评分 |
| `evaluation/src/check_run_info.py` | run_info 完整性、哈希与证据链核查 |
| `evaluation/src/compare_evaluations.py` | 多模型、多方法对比 |
| `evaluation/src/export_errors.py` | 错误样例导出 |
| `evaluation/docs/` | 交接规范、评分口径、平台规则与前期核验材料 |
| `evaluation/examples/` | 人工接口样例与流水线验收样例 |

详细说明见 [`evaluation/README.md`](../evaluation/README.md)。

## `doc/`

| 文件 | 内容 |
|---|---|
| `BERT数据接口说明_v2.md` | 给成员 B 的数据接口契约 |
| `实验记录.md` | 七次实验的配置、指标、耗时（脚本自动生成） |
| `耗时参考.md` | 各任务在 本地 GPU / 云服务器上的实测耗时 |
| `项目状态.md` | 进度、待确认事项 |
| `附录A_复现指南.md` `附录B_代码结构说明.md` | 本附录 |

## 数据与产物约定

- 原始数据、模型权重、Checkpoint、虚拟环境**不入库**。
- `runs/<run_id>/` 保存配置、训练日志、逐轮指标；权重在 `checkpoints/`（已忽略）。
- 每次实验的预测与 run_info 在 `runs/<run_id>/predict_<split>/`。
