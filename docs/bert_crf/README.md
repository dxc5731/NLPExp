# BERT-CRF实验与证据索引

任务：IMCS-V2-NER / Task1中文医疗命名实体识别；字符级原始BIO监督，BERT→offset_share_unique_v1→Linear→严格BIO约束CRF。

| 实验 | 状态 | 最佳Epoch | 内部val严格Micro F1 | 记录 |
| --- | --- | --- | --- | --- |
| Baseline Run 1 | 完成5轮，中途停止后从Epoch2恢复 | 5 | **89.6837%** | [正式实验记录](experiments/baseline_run1.md) · [结构化摘要](experiments/baseline_run1_summary.json) · [公开错误分析](analysis/baseline_run1_error_analysis.md) |
| Experiment 2：输出头Dropout=0.2 | 完成5轮；E5末同时达到早停条件和轮数上限 | 3 | **89.8623%（+0.1786 pp）** | [完整对照报告](experiments/dropout02_vs_baseline.md) · [结构化摘要](experiments/dropout02_summary.json) · [完整数值数据](analysis/dropout02_comparison_data.json) |

- [正式配置](../../configs/bert_crf/first_baseline.json)：首轮原始配置，输出头Dropout0.1，保持不变。
- 训练代码提交：`a955f56855f72c944010b93c9965c01187c15b4d`；配置身份、数据/划分/Checkpoint哈希和换行字节差异见实验记录及摘要。
- [训练曲线](figures/baseline_training_curves.png)、[错误分布图](figures/baseline_error_patterns.png)、[完整聚合统计CSV](analysis/grouped_error_statistics.csv)。
- 工程阶段材料：[模型核心](02_model_core_and_smoke_tests.md)、[训练框架](03_training_framework.md)、[字符适配](05_character_adapter_integration.md)、[配置决定](07_baseline_config_decisions.md)、[启动前验收](08_training_preflight_and_integration.md)。

本次GitHub归档只保存公开适用的聚合统计、方法、图表和结论。完整基准审计在本地`docs/bert_crf/evidence/20261020T230513+0800_baseline_run1_audit/`，错误分析在本地`docs/bert_crf/evidence/20261020T234238+0800_baseline_error_analysis/`；这些被忽略的目录不是GitHub下载链接。原文、完整内部预测/错误、23例文本、模型/优化器权重、敏感日志继续本地受控保存，不自动加入课程最终ZIP。匿名样本ID和哈希不构成公开许可。

后续实验必须使用独立运行目录，保存实际配置、Git版本、环境、日志、预测、指标、Checkpoint身份和必要哈希；中止/恢复分别留痕，禁止覆盖旧证据。数据、固定划分与权重未公开，仅凭GitHub文档不能完整复现训练，需要获得合规访问并匹配记录版本。

Baseline归档时记录的计划（历史内容，第二轮现已完成）：

下一步：先人工复核实体边界与类别口径，再视情况开展仅输出头Dropout0.1→0.2的单因素对照；学习率、数据、映射、轮数、Batch、种子及其他设置保持一致。具体收益尚未验证，本次归档不启动训练或修改正式配置。

第二轮已完成：只改变输出头Dropout0.1→0.2，BERT内部Dropout仍0.1，训练源码提交`7c9b4d7e45398de445e019772fc8e690aa3099e8`。真实配置见[dropout_02.json](../../configs/bert_crf/dropout_02.json)，保留原资源证据引用和哈希；未公开的本地数据、划分、权重等须另行准备。

- [Loss曲线](figures/dropout02_loss_comparison.png) · [F1及P/R曲线](figures/dropout02_f1_comparison.png) · [五类实体F1](figures/dropout02_entity_f1_comparison.png) · [错误分量对比](figures/dropout02_error_comparison.png)。
- 当前观察：MicroF1提高0.1786pp，主要来自FP减少；Recall及三类实体F1下降，后期验证NLL上升未解决。保留0.2/E3作为候选，同时保留Baseline，不宣称稳定普遍最优。
- 第二轮完整审计本地位置：`docs/bert_crf/evidence/20261021T155410+0800_dropout02_result_audit/`；完整训练与敏感证据继续本地受控保存。
- 当前建议：先人工复核Operation新增过长边界与Medical漏识别/错类；之后视目标考虑固定其他条件的学习率2e-5→1e-5，或配对多seed验证Dropout稳定性。本次只整理、提交和同步成果，不执行训练。


## 第三轮已审计：学习率 1e-5，以及三轮总结

本节更新当前候选建议，前文前两轮的记录与当时结论作为历史保留。

| 实验 | LR | 输出头Dropout | 最佳Epoch | Precision | Recall | Micro F1 | Macro F1 | 相对Baseline F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Experiment 3：学习率对照 | 1e-5 | 0.1 | 5 | 89.4847% | 90.5014% | **89.9902%** | 89.7347% | **+0.3065 pp** |

- [完整第三轮审计与三轮对照报告](experiments/lr1e5_vs_previous.md) · [第三轮结构化摘要](experiments/lr1e5_summary.json) · [三轮完整聚合数值数据](analysis/three_experiments_comparison.json) · [实际正式配置](../../configs/bert_crf/lr_1e-5.json)。
- [三轮Train/Val NLL](figures/three_runs_loss_comparison.png) · [Micro F1与P/R](figures/three_runs_f1_comparison.png) · [五类实体F1](figures/three_runs_entity_f1_comparison.png) · [全部错误分量](figures/three_runs_error_comparison.png)。
- 第三轮正式run：`20261021T195736+0800_real_training_95abb833`；正常完成5轮，27765次更新，fit5689.894秒，无恢复或重启。最佳Checkpoint SHA-256：`e5ee6aaa8d4a00a4626553002aa9a63253a8ee0139463fec2bb41427bf391bed`。
- 单变量核验通过：相对Baseline只将统一BERT/Linear/CRF学习率2e-5→1e-5，输出头及BERT内部Dropout仍0.1；数据、映射、CRF、Scheduler、Warmup及选模规则保持一致。
- 当前内部Micro/Macro/Recall最高候选为**LR1e-5/E5**；Precision最高仍是Dropout0.2/E3。第三轮P/R同时提高，FP-22、FN-19，但Drug与Medical_Examination退化。Val NLL后期上升缓解而未消失。
- 对话级10000次Bootstrap（seed20261021）的第三轮−Baseline Delta F1探索性95%区间为[-0.0601,+0.6744] pp，跨零。单seed、同一内部val选模，不证明稳定最优；第二轮与第三轮不是单变量对照。
- 建议先冻结三份已审计候选并准备小组交接；有预算时优先配对多seed验证Baseline与LR1e-5，而非立即开展广泛调参或默认组合Dropout0.2更好。本轮没有执行新训练或官方dev/test预测。
- 完整本地审计：`docs/bert_crf/evidence/20261021T214932+0800_lr1e5_result_audit/`；归档过程：`docs/bert_crf/evidence/20261021T214932+0800_lr1e5_github_archive/`；原始训练：`experiments/bert_crf/20261021T195736+0800_real_training_95abb833/`。这些被忽略目录仅为本地索引，不是GitHub下载链接。原始数据、逐句预测、Checkpoint、恢复状态和敏感日志继续本地受控保存。


## 第四轮已审计：边界辅助学习与四轮对照

前三轮记录及当时结论保留；本节为当前结论。

| 实验 | LR | Dropout | 边界辅助λ | 最佳Epoch | Precision | Recall | Micro F1 | Macro F1 | ΔF1 vs Baseline / 第三轮 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Boundary Auxiliary | 1e-5 | 0.1 | 0.2 | 3 | 89.8292% | 90.0470% | **89.9380%** | 89.5896% | +0.2542 / −0.0522 pp |

- [独立审计报告](experiments/boundary_aux_run1.md) · [四轮完整对照](experiments/four_runs_comparison.md) · [摘要](experiments/boundary_aux_summary.json) · [完整聚合JSON](analysis/four_runs_comparison.json) · [全标量CSV](analysis/four_runs_comparison.csv) · [真实配置](../../configs/bert_crf/boundary_aux_02.json)。
- [CRF NLL](figures/four_runs_loss_comparison.png) · [Micro F1](figures/four_runs_f1_comparison.png) · [五类实体](figures/four_runs_entity_f1_comparison.png) · [边界/错误分量](figures/four_runs_error_comparison.png) · [配对变化](figures/boundary_aux_paired_changes.png) · [辅助/Total Loss](figures/boundary_aux_losses.png)。
- run `20261021T231741+0800_real_training_86767c7e`，5轮27765次更新、exit0，E5末early_stopped；最佳E3 SHA-256 `e06bae70f7030dd21237578df6e68d9d2f225c1d54781edec326ee196c94a307`。四轮20份逐Epoch预测复算通过，第四轮5份Checkpoint可读、最佳完整9687句重新推理一致，267项测试通过。
- 实际新增Start/End各Linear(768,1)，原字符BIO生成目标，平衡BCE，total=CRF+.2×boundary，最终仍CRF；训练源码快照/当前源码逐字节吻合。实际实现与必要测试一同归档，未改训练后的逻辑或历史配置。
- 相对第三轮：一对一同类型边界272→263、Medical42→35；过长153→155、纯漏215→237、合并19→26。FP−29/FN+30；Medical F1+0.5213pp，另外四类下降。247对话/10000次配对Bootstrap的DeltaF1区间[-0.4256,+0.3238]pp跨零，不能声称稳定优势。
- 当前主候选仍**第三轮LR1e-5/E5**（Micro/Macro/Recall最高）；第四轮Precision最高。先冻结交接、复核短实体漏检与合并，再视预算考虑配对多seed；不自动决定新参数或第五轮。
- 完整审计本地`docs/bert_crf/evidence/20261022T200005+0800_boundary_aux_result_audit`，归档过程为其`publication/`；启动证据`docs/bert_crf/evidence/20261021T224813+0800_boundary_aux_launch`，原训练`experiments/bert_crf/20261021T231741+0800_real_training_86767c7e`。这些为本地受控索引，不是GitHub下载链接。原始文本、完整预测/错误、Checkpoint、敏感日志、权重与资源文件不公开。
