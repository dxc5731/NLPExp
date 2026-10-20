# NLPExp

IMCS-V2-NER / Task1 课程项目。`dxc` 分支包含个人负责的字符级 BERT-CRF 与阶段 0—1E 的工程记录。

## 当前进度

已实现 BERT 编码器、字符级 Dropout/Linear/CRF、训练和解码一致的 BIO 硬约束、输入校验与保存重载。阶段 1B 完整验收 **53 passed，0 failed，0 skipped**，包括真实预训练 BERT 的 GPU 前反向及 Checkpoint 一致性。

阶段 1C 新增训练器、严格字符/实体指标、运行记录、最佳模型和 Epoch 边界恢复。完整回归 **130 passed，0 failed，0 skipped**；真实 BERT 在 4 条人工样例上完成 3 个短 Epoch，轻量 BERT 的 CPU/CUDA 连续运行与恢复状态逐位一致。

阶段 1D 完成公共接口 v2 审计和真实字符适配，全量 98529 句/1279361 字符覆盖通过。阶段 1E 接通受保护的正式入口和独立配置，完整回归 **239 passed，0 failed，0 skipped**；真实无更新 Preflight、资源压力测试及隔离人工训练/恢复通过。

**正式真实 Epoch 尚未启动，没有正式 F1 或 dev/test 预测。** 首次配置是待本人审核并授权运行的建议值；人工 Smoke Loss 和指标不代表医疗模型效果。

## 代码与证据

- 核心：[`src/bert_crf/`](src/bert_crf/)
- 测试：[`tests/bert_crf/`](tests/bert_crf/)
- 配置：[`configs/bert_crf/stage1b_core_smoke.json`](configs/bert_crf/stage1b_core_smoke.json)
- [阶段 0 审查](docs/bert_crf/00_initial_audit.md)、[阶段 1A 环境与接口实测](docs/bert_crf/01_environment_and_dataloader.md)、[对齐审计](docs/bert_crf/alignment_summary.md)
- [阶段 1B 实现、命令与验收详情](docs/bert_crf/02_model_core_and_smoke_tests.md)
- [阶段 1C 训练框架、人工测试、恢复及复现详情](docs/bert_crf/03_training_framework.md)
- [阶段 1C 人工测试配置](configs/bert_crf/stage1c_trainer_smoke.json)
- [阶段 1D-0 接口审计](docs/bert_crf/04_interface_v2_audit.md)、[字符适配与监督差异](docs/bert_crf/05_character_adapter_integration.md)
- [阶段 1D 收尾阻塞记录](docs/bert_crf/06_pretraining_readiness_review.md)、[1E 配置决定依据](docs/bert_crf/07_baseline_config_decisions.md)
- [阶段 1E 验收、完整命令与产物说明](docs/bert_crf/08_training_preflight_and_integration.md)、[首次正式配置](configs/bert_crf/first_baseline.json)
- [已核对的真实开发日志与结构化证据](docs/bert_crf/published_evidence/README.md)

## 轻量复查

已验收环境：Windows 11、Python 3.13.5、PyTorch 2.14.1+cu132、Transformers 5.20.1、RTX 5060 Laptop；完整版本见 [依赖清单](docs/bert_crf/requirements-stage1a.lock.txt)。安装与官方 CUDA wheel 来源见阶段 1A 文档，避免向 base 安装。其他环境未在本阶段验收。

项目根目录，使用已配置的隔离环境：

```text
.venv-bert-crf/Scripts/python.exe -X utf8 -m pytest tests/bert_crf -q --basetemp=.local/bert_crf/unit-recheck
```

该轻量命令不隐式读取真实数据，会明确跳过未提供环境的集成项。完整 239 项验收命令及真实/人工证据路径见阶段 1E 文档，每次重跑使用新结果和 Checkpoint 路径。历史 Smoke seed=1729，首次基准候选 seed=42，均为 float32。

## 正式启动前检查（不训练）

在项目根目录，复用本机已审计的 `.local` 数据/接口/权重路径：

```text
.venv-bert-crf/Scripts/python.exe -X utf8 scripts/train_bert_crf.py --config configs/bert_crf/first_baseline.json --preflight
```

此命令无需手工设置 PYTHONPATH，默认创建唯一的本地检查目录，检查全量数据并执行真实 GPU 无更新前后向。正式训练必须另行显式选择 `--train`，并等待本人审核配置和授权。配置绑定数据/模型/Tokenizer/资源证据哈希；克隆到新机器后须准备匹配的本地文件并复核环境/资源，不会自动下载、改路径或回退人工数据。

原始医疗数据、ZIP、模型权重、Checkpoint、虚拟环境和敏感日志不纳入仓库。开发证据中保留的本地绝对路径只是历史定位，克隆后应按当前目录配置。AI 辅助范围及程序验证方式已在各阶段记录，正式实验设计、结果分析与个人贡献声明仍由组员审核完成。
