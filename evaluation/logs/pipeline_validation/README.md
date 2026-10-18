# 第三阶段实跑证据（非模型实验）

2026-10-18 首次发布说明：本目录选取下面链接的最终历史验收，日志／JSON／CSV 保留原始字节及本机执行路径；中间 `batches/` 和另一 acceptance 目录仅在原工作目录留存。仓库迁移后的新验证见[重新检出验证](../repository_validation/README.md)。

2026-10-17，Windows，Python 3.14.0，标准库 CPU。此处全部评分都是 `synthetic_integration_test`，没有真实训练模型、推理耗时或正式成绩。历史批次保留不覆盖。

最终验收索引：[acceptance_summary.json](acceptance/20261017T104721155709_pb8d_vw9/acceptance_summary.json)。各命令、退出码、墙钟耗时及完整控制台输出在同目录。

| 检查 | 实际结果 | 日志 |
|---|---|---|
| 第一阶段回归 | 42/42，0 失败，exit 0 | [stage1.txt](acceptance/20261017T104721155709_pb8d_vw9/stage1.txt) |
| 第二阶段回归 | 47/47，0 失败，exit 0 | [stage2.txt](acceptance/20261017T104721155709_pb8d_vw9/stage2.txt) |
| 第三阶段新增 | 49/49，0 失败，exit 0 | [stage3.txt](acceptance/20261017T104721155709_pb8d_vw9/stage3.txt) |
| 全量 unittest | 138/138，0 失败，exit 0 | [all.txt](acceptance/20261017T104721155709_pb8d_vw9/all.txt) |
| 三方法完整批量 CLI | SUCCESS，self_check=PASS，exit 0 | [three_method_cli.txt](acceptance/20261017T104721155709_pb8d_vw9/three_method_cli.txt) |
| 官方完整 dev 只读审计 | PASS，833 对话／33267 句／441339 码点，0 错误 | [审计 JSON](acceptance/20261017T104721155709_pb8d_vw9/official_dev_readonly.json) |

可重复执行（在项目根目录）：

```powershell
python -X utf8 tests/run_pipeline_acceptance.py
```

会新建证据子目录。内部调用分阶段及全量 unittest、一条批量 CLI、官方 dev 校验；独立调用既有评分函数核对各方法指标，复读错误 CSV、比较 CSV／JSON，核查输入文件哈希未变。完整命令行参数保存在 summary 的 commands 数组。若失败则终止并保留已产生的 transcript，不写 PASS 总结。

## 三方法彩排的实际产物

[批次总报告](acceptance/20261017T104721155709_pb8d_vw9/batches/20261017T104739822386_4o3zgaqq/batch_report.json)；[整体 CSV](acceptance/20261017T104721155709_pb8d_vw9/batches/20261017T104739822386_4o3zgaqq/comparison/model_comparison.csv)；[五类 CSV](acceptance/20261017T104721155709_pb8d_vw9/batches/20261017T104739822386_4o3zgaqq/comparison/entity_type_comparison.csv)；[完整比较 JSON](acceptance/20261017T104721155709_pb8d_vw9/batches/20261017T104739822386_4o3zgaqq/comparison/comparison.json)。

| 人工交接身份 | TP/FP/FN | 期望 F1 | 实际 F1 | 错误表 FP/FN 行 |
|---|---|---|---|---|
| synthetic_rules_v1 | 0/0/5 | 0 | 0 | 0/5 |
| synthetic_crf_v1 | 2/2/3 | 4/9 | 0.4444444444444444 | 2/3 |
| synthetic_bert_v1 | 4/1/1 | 4/5 | 0.8 | 1/1 |

每方法保留 `runs/001`、`002`、`003` 下的 evaluation.json、reproducibility_check.json、run_status.json，以及 errors 下 entity_errors.csv、entity_matches.csv、error_candidates.csv、error_summary.json。默认未启用 local-text，产物不含完整医疗原文。

人工 gold 共 1 对话／2 句／22 字符，五类各 1 个实体；三方法共 6 条预测句子记录，只有 2 条不同句子。三份预测全部校验 PASS，元数据的 synthetic 专用检查 PASS；该状态不表明真实模型已复现。各自实体、字符指标与独立评分完全相同，五类计数和整体 gold=TP+FN、predicted=TP+FP 一致。两个比较 CSV 全部单元格复读一致。候选一对一且不更改原始 FP/FN。

本次批量进程墙钟约 0.387 秒，程序记录流水线耗时约 0.177 秒；仅为 2 句小样例的工程计时，**不是模型推理速度，也不是完整 dev 批量性能测试**。

异常分支通过自动化测试实际执行：格式／哈希／元数据缺失、重复 ID、test 无 gold、数据版本／gold／政策不兼容、人工混入真实比较、部分失败、同方法多 run、不覆盖历史、中文及 Windows／相对路径、BOM 与公式防护、歧义不强配对、保存文件篡改和最终自检失败。没有模拟真实模型训练的成功结论。

## 官方数据与限制

官方 ZIP 原地只读；核对原始 JSON 字段、BIO、字符对齐和完整规模，未生成官方数据预测。原 ZIP SHA-256 前后均为 `84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67`；dev 成员原始字节 SHA-256 为 `16f597ad368971c7db994ca904e05d91942f0e3c73009c9ef1d597fde065d39f`。人工输入哈希同样未变。

未做完整 dev 的批量评分／压力测试，未实测不同 Python 版本，尚待真实预测验证成员自己的数据对齐、运行证据和复跑过程。模型身份、无泄漏、子词映射、复现与耗时真实性必须人工确认。程序只输出差异候选，不进行正式根因分析、实验设计或课程结论；未制作或上传天池 ZIP。

早期 `batches/` 和另一 acceptance 目录属于本轮中间验证，保留历史；最终验收以本页链接的 138 项版本为准。
