# NLPExp

IMCS-V2-NER / Task1 课程项目。`dxc` 分支包含个人负责的字符级 BERT-CRF 核心与阶段 0—1B 的开发记录。

## 当前进度

已实现 BERT 编码器、字符级 Dropout/Linear/CRF、训练和解码一致的 BIO 硬约束、输入校验与保存重载。阶段 1B 完整验收 **53 passed，0 failed，0 skipped**，包括真实预训练 BERT 的 GPU 前反向及 Checkpoint 一致性。

真实 Token→字符适配仅定义契约，尚待公共接口修复后集成；没有正式训练、实体级 F1 或 dev/test 预测。人工 Smoke Loss 不代表训练效果。

## 代码与证据

- 核心：[`src/bert_crf/`](src/bert_crf/)
- 测试：[`tests/bert_crf/`](tests/bert_crf/)
- 配置：[`configs/bert_crf/stage1b_core_smoke.json`](configs/bert_crf/stage1b_core_smoke.json)
- [阶段 0 审查](docs/bert_crf/00_initial_audit.md)、[阶段 1A 环境与接口实测](docs/bert_crf/01_environment_and_dataloader.md)、[对齐审计](docs/bert_crf/alignment_summary.md)
- [阶段 1B 实现、命令与验收详情](docs/bert_crf/02_model_core_and_smoke_tests.md)
- [已核对的真实开发日志与结构化证据](docs/bert_crf/published_evidence/README.md)

## 轻量复查

已验收环境：Windows 11、Python 3.13.5、PyTorch 2.14.1+cu132、Transformers 5.20.1、RTX 5060 Laptop；完整版本见 [依赖清单](docs/bert_crf/requirements-stage1a.lock.txt)。安装与官方 CUDA wheel 来源见阶段 1A 文档，避免向 base 安装。其他环境未在本阶段验收。

项目根目录，使用已配置的隔离环境：

```text
.venv-bert-crf/Scripts/python.exe -X utf8 -m pytest tests/bert_crf -q --basetemp=.local/bert_crf/unit-recheck
```

预期在该 CUDA 环境为 52 passed、1 skipped：该命令未传真实权重目录，因此明确跳过预训练集成项。完整 53 项 GPU 验收需要固定 revision 的实际权重及显式本地证据路径，命令见阶段 1B 文档；每次重跑应使用新的结果和 Checkpoint 路径。固定 seed 为 1729，配置为 float32，未运行优化器更新。

原始医疗数据、ZIP、模型权重、Checkpoint、虚拟环境和敏感日志不纳入仓库。开发证据中保留的本地绝对路径只是历史定位，克隆后应按当前目录配置。AI 辅助范围及程序验证方式已在各阶段记录，正式实验设计、结果分析与个人贡献声明仍由组员审核完成。
