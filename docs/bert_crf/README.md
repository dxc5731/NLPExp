# BERT-CRF实验与证据索引

任务：IMCS-V2-NER / Task1中文医疗命名实体识别；字符级原始BIO监督，BERT→offset_share_unique_v1→Linear→严格BIO约束CRF。

| 实验 | 状态 | 最佳Epoch | 内部val严格Micro F1 | 记录 |
| --- | --- | --- | --- | --- |
| Baseline Run 1 | 完成5轮，中途停止后从Epoch2恢复 | 5 | **89.6837%** | [正式实验记录](experiments/baseline_run1.md) · [结构化摘要](experiments/baseline_run1_summary.json) · [公开错误分析](analysis/baseline_run1_error_analysis.md) |

- [正式配置](../../configs/bert_crf/first_baseline.json)：首轮原始配置，输出头Dropout0.1，保持不变。
- 训练代码提交：`a955f56855f72c944010b93c9965c01187c15b4d`；配置身份、数据/划分/Checkpoint哈希和换行字节差异见实验记录及摘要。
- [训练曲线](figures/baseline_training_curves.png)、[错误分布图](figures/baseline_error_patterns.png)、[完整聚合统计CSV](analysis/grouped_error_statistics.csv)。
- 工程阶段材料：[模型核心](02_model_core_and_smoke_tests.md)、[训练框架](03_training_framework.md)、[字符适配](05_character_adapter_integration.md)、[配置决定](07_baseline_config_decisions.md)、[启动前验收](08_training_preflight_and_integration.md)。

本次GitHub归档只保存公开适用的聚合统计、方法、图表和结论。完整基准审计在本地`docs/bert_crf/evidence/20261020T230513+0800_baseline_run1_audit/`，错误分析在本地`docs/bert_crf/evidence/20261020T234238+0800_baseline_error_analysis/`；这些被忽略的目录不是GitHub下载链接。原文、完整内部预测/错误、23例文本、模型/优化器权重、敏感日志继续本地受控保存，不自动加入课程最终ZIP。匿名样本ID和哈希不构成公开许可。

后续实验必须使用独立运行目录，保存实际配置、Git版本、环境、日志、预测、指标、Checkpoint身份和必要哈希；中止/恢复分别留痕，禁止覆盖旧证据。数据、固定划分与权重未公开，仅凭GitHub文档不能完整复现训练，需要获得合规访问并匹配记录版本。

下一步：先人工复核实体边界与类别口径，再视情况开展仅输出头Dropout0.1→0.2的单因素对照；学习率、数据、映射、轮数、Batch、种子及其他设置保持一致。具体收益尚未验证，本次归档不启动训练或修改正式配置。
