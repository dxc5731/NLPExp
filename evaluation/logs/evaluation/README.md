# 第二阶段评分器验证证据（2026-10-17）

**本目录全部数值属于人工数据上的评分器验证，不能作为本组模型实验成绩。** 没有训练／推理、真实预测、正式错误归因、天池提交或付费服务。实跑环境 Windows、Python 3.14.0；运行程序仅依赖标准库，未逐一在其他 Python 3.10+ 版本测试。

| 证据 | 实际结果 |
|---|---|
| [stage1_regression_20261017.txt](stage1_regression_20261017.txt) | 第一阶段原有 42 项测试全部通过、0 失败；未修改原测试文件 |
| [stage2_unit_tests_20261017.txt](stage2_unit_tests_20261017.txt) | 第二阶段 47 项测试全部通过、0 失败；浮点绝对容差 1e-12 |
| [manual_cli_20261017.txt](manual_cli_20261017.txt) | 人工样例完整 CLI，退出码 0，SCORED；包含实际命令和生成结果的完整路径 |
| [人工 CLI 结果](manual_cli/20261017T100936263720_xe7oc7wq/evaluation.json) | purpose=scorer_validation，1 对话／2 句／22 字符，验证 PASS，gold 实体 5／预测 4，实体 TP/FP/FN=2/2/3，Micro-P=0.5、R=0.4、F1=4/9 |
| [author_crosscheck_20261017.json](author_crosscheck_20261017.json) | 四组合法人工场景，作者实际实体代码段 vs 本工具：overall 及五类 F1 全部通过；详见范围说明 |
| [data_audit_dev_20261017.json](data_audit_dev_20261017.json) | 官方 ZIP 中 dev 只读结构核查 PASS，833 对话／33267 句／441339 字符，错误 0；没有对真实数据评分 |

两阶段共 89 项测试，全部通过。非法输入测试的“通过”指正确拒绝非法输入，不是对其放行。测试还验证拒绝时不调用评分核心、不产生含指标的 evaluation.json，test 无 gold 时不能评分，NOT_RUN 不放行，输出重复执行不覆盖，字符统计保留 O 与实体之间的误报／漏报。

人工 CLI 的字符手算与实际一致：正确 17/22；含 O 的 micro-P/R/F1=17/22、macro-F1=53/99；不含 O 的 micro-P=5/7、R=1/2、F1=10/17、macro-F1=1/2。包含／排除 O 的 macro 分母固定为 11／10，缺席类为 0。原始计数和混淆矩阵均在 JSON 内，可复核。

作者交叉验证四组 overall F1 依次为 4/9、1、0、10/11。原始 overall 容差 1e-12；每类与作者四位小数输出的容差为 0.000050000001。舍入样例包含 Symptom F1=2/3 对作者 0.6667；混合样例 overall 0.4444444444444444 对作者 0.4444444444444445 是等价公式的浮点尾差。

固定版本源码已于本轮重新在线获取，与本地 `docs/前期核验/eval_task1_upstream.py` 字节一致，SHA-256 为 `9af0a1bdbdc813ff8f60b149c41e32d0d0b96f0cecb32b2bf0e37d568563d919`。实际执行的是 AST 提取的原始 get_entity_bio、compute 和 ner_eval 实体段；本机缺少 seqeval，**未运行完整作者脚本／CLI、token 段或天池服务器**，也不拿 token 段与字符指标作对照。五类金标准均有实例；作者 gold 类别缺席时的 KeyError 不是本工具的限制。

原始 ZIP 哈希仍为 `84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67`，结构报告记录相同哈希；人工输入哈希与三个实现文件哈希记录在 CLI 结果中。本目录所有报告均不输出真实医疗原句。第一阶段历史证据未覆盖。

复跑命令及报告字段见 [统一评分说明](../../docs/统一评分说明.md)。CLI 总会创建新目录，故复跑路径不同；作者交叉报告独占创建，重复运行需要换报告文件名。当前仍未核验真实模型输出、run_info 真伪、tokenizer 对齐忠实性及训练复现，未固定团队实验设置。
