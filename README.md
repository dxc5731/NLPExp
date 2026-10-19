# NLPExp

IMCS-V2-NER / Task1 课程项目。`dxc` 分支包含个人负责的字符级 BERT-CRF 核心与阶段 0—1C 的开发记录。

## 当前进度

已实现 BERT 编码器、字符级 Dropout/Linear/CRF、训练和解码一致的 BIO 硬约束、输入校验与保存重载。阶段 1B 完整验收 **53 passed，0 failed，0 skipped**，包括真实预训练 BERT 的 GPU 前反向及 Checkpoint 一致性。

阶段 1C 新增训练器、严格字符/实体指标、运行记录、最佳模型和 Epoch 边界恢复。完整回归 **130 passed，0 failed，0 skipped**；真实 BERT 在 4 条人工样例上完成 3 个短 Epoch，轻量 BERT 的 CPU/CUDA 连续运行与恢复状态逐位一致。

真实 Token→字符适配仅定义契约，尚待公共接口修复后集成；正式训练入口保持关闭，没有真实数据微调、正式 F1 或 dev/test 预测。人工 Smoke Loss 和指标不代表医疗模型效果。

## 代码与证据

- 核心：[`src/bert_crf/`](src/bert_crf/)
- 测试：[`tests/bert_crf/`](tests/bert_crf/)
- 配置：[`configs/bert_crf/stage1b_core_smoke.json`](configs/bert_crf/stage1b_core_smoke.json)
- [阶段 0 审查](docs/bert_crf/00_initial_audit.md)、[阶段 1A 环境与接口实测](docs/bert_crf/01_environment_and_dataloader.md)、[对齐审计](docs/bert_crf/alignment_summary.md)
- [阶段 1B 实现、命令与验收详情](docs/bert_crf/02_model_core_and_smoke_tests.md)
- [阶段 1C 训练框架、人工测试、恢复及复现详情](docs/bert_crf/03_training_framework.md)
- [阶段 1C 人工测试配置](configs/bert_crf/stage1c_trainer_smoke.json)
- [已核对的真实开发日志与结构化证据](docs/bert_crf/published_evidence/README.md)

## 轻量复查

已验收环境：Windows 11、Python 3.13.5、PyTorch 2.14.1+cu132、Transformers 5.20.1、RTX 5060 Laptop；完整版本见 [依赖清单](docs/bert_crf/requirements-stage1a.lock.txt)。安装与官方 CUDA wheel 来源见阶段 1A 文档，避免向 base 安装。其他环境未在本阶段验收。

项目根目录，使用已配置的隔离环境：

```text
.venv-bert-crf/Scripts/python.exe -X utf8 -m pytest tests/bert_crf -q --basetemp=.local/bert_crf/unit-recheck
```

该命令未传真实权重目录和人工训练结果，明确跳过两项集成测试。完整 130 项验收命令见阶段 1C 文档，包含固定 revision 真实权重及人工训练器运行；每次重跑应使用新的结果和 Checkpoint 路径。固定 seed 为 1729，配置为 float32；阶段 1C 的优化器更新仅针对明确标注的人工数据。

原始医疗数据、ZIP、模型权重、Checkpoint、虚拟环境和敏感日志不纳入仓库。开发证据中保留的本地绝对路径只是历史定位，克隆后应按当前目录配置。AI 辅助范围及程序验证方式已在各阶段记录，正式实验设计、结果分析与个人贡献声明仍由组员审核完成。
