# 中文医患对话医疗实体识别

本页是仓库 `evaluation/` 模块说明。以下命令均在该目录运行；从仓库根目录先执行 `cd evaluation`。前期本地工作于 2026-10-18 首次整理入库，历史日志保留原始字节与原执行日期，迁入及修改清单见[仓库发布说明](docs/仓库发布说明.md)。

## 首次检出的准备与验证

Python 3.10+，仅标准库、CPU；本次实际验证版本会写入新验收日志。人工验证不需要数据集或模型。先准备测试用的固定作者参考，再执行：

```powershell
python -X utf8 tests/prepare_author_reference.py
python -X utf8 -m unittest discover -s tests -v
python -X utf8 src/run_evaluation_batch.py --manifest examples/pipeline_validation/evaluation_runs.json --csv-bom
```

参考准备命令首次需访问固定 raw URL，校验 SHA-256 后保存至被 Git 忽略的 `docs/前期核验/eval_task1_upstream.py`。上游未声明许可证，本仓库仅提供出处和下载准备工具，不分发该作者源码。离线已有相同副本时可运行 `python -X utf8 tests/prepare_author_reference.py --source <副本路径>`；原有文件哈希不符时拒绝替换。参考源码只用于测试实体代码段，运行评估工具本身不需要它。

预期 138 项现有测试通过；人工批量命令输出 `SUCCESS` 与 `self_check=PASS`。输出在 `logs/pipeline_validation/batches/`，该运行目录默认不提交。

真实平台数据从[天池数据入口](https://tianchi.aliyun.com/dataset/95414)按权限自行获取，放到 `data/IMCS-V2-NER.zip` 或在命令中传入实际位置，文件不进 Git。固定哈希及说明见[data/README.md](data/README.md)。官方包就绪后，可完整重跑分阶段、全量、人工三方法与官方 dev 只读审计：

```powershell
python -X utf8 tests/run_pipeline_acceptance.py --official-data data/IMCS-V2-NER.zip
```

该命令只用人工数据评分，对完整官方 dev 只读核验结构、BIO 和字符对应，不运行真实模型。

自然语言处理课程项目，方向 **B：信息抽取／序列标注**。当前使用 IMCS-V2-NER（Task1），对医患对话逐句预测字符级 BIO 标签，识别症状、药品名、药物类别、检查和医疗操作五类实体。

## 当前状态

已完成前三阶段评估工程：统一预测接口、严格实体及真正逐字符评分，以及错误导出、复现证据核查、多模型比较和一键批量接入。尚无本组真实模型预测／成绩，未正式提交天池；人工样例数值单独留作工程验证，不是模型实验。下一步接入队友真实预测，不继续扩展非必要功能。

| 分工 | 责任 | 后续产出 |
|---|---|---|
| 成员 1、2 | BERT 微调主方案 | 训练／预测代码、配置、真实日志、预测文件 |
| 成员 3 | 规则／训练集词典与独立 CRF 基线 | 基线代码、配置、真实日志、预测文件 |
| 用户 | 统一评估、方法比较、复现、天池对接 | 口径与校验代码、可复算结果表、复现及提交记录 |

成员姓名、模型版本与具体实验设置待团队补充。各成员共同核对方案，独立完成实验设计、结果解释及各自贡献声明。

## 从哪里开始

- [AI交互记录](AI交互记录.md)：选型、人工决策、AI 操作、调整原因及产出；后续每轮追加。
- [课程要求核对](docs/课程要求核对.md)：2026-10-16 对任务书的重新核对。
- [本地评估与天池提交说明](docs/本地评估与天池提交说明.md)：2026-10-16 再次查证的计分口径、评估职责与两类提交要求。
- [预测结果交接规范](docs/预测结果交接规范.md)：可发给三位模型同学的统一接口、元数据要求、校验命令及交接清单。
- [统一评分说明](docs/统一评分说明.md)：实体／字符口径、评分命令、输出结构与作者实体评分交叉验证边界。
- [真实模型预测接入与评估指南](docs/真实模型预测接入与评估指南.md)：真实交付、配置、单命令、错误定位与人工复核边界。
- [第三阶段人工彩排与证据](logs/pipeline_validation/README.md)：三方法完整接入和回归测试；非真实成绩。
- [完全人工构造的接口示例](examples/interface_example/README.md)：原文、预测 JSON 与 run_info 示例，不能当作模型结果。
- [人工评分器验证样例](examples/scorer_validation/README.md)：有明确手算期望值的五类混合样例；[第二阶段证据](logs/evaluation/README.md) 留存实跑结果。
- [前期核验资料](docs/前期核验/README.md)：2026-10-15 的数据、平台与评估证据，历史快照不代表当前榜单。
- [资料来源清单](docs/资料来源.json)：任务书 URL／获取时间、迁入资料原路径及 SHA-256。

```text
evaluation/
├── AI交互记录.md
├── AGENTS.md                 # 编码与持续记录约定
├── README.md
├── docs/                     # 课程快照和前期核验资料
├── src/                      # 校验、纯指标、评分、错误导出、元数据、比较与批量入口
├── tests/                    # 人工校验／评分测试及作者实体段对照
├── examples/interface_example/ # 人工接口示例，不是模型输出
├── examples/scorer_validation/ # 人工评分器样例，不是模型实验
├── examples/pipeline_validation/ # 三方法人工交接，非真实预测
├── configs/                  # 批量清单及 run_info 待填写模板
├── logs/                     # 校验证据及后续真实运行日志，分目录保存
├── results/                  # 后续汇总指标与对照材料
└── data/                     # 本机数据位置；原始数据不提交
```

`results/` 尚无真实成绩；`logs/validation/`、`logs/evaluation/` 和 `logs/pipeline_validation/` 存分阶段工程测试及只读数据核查证据，不含模型实验成绩。

## 第一阶段：如何运行输入校验

Python 3.10+，仅标准库、CPU 即可。以下命令在评估模块根目录（仓库中的 `evaluation/`）执行。

```powershell
python -X utf8 -m unittest discover -s tests -v
python -X utf8 src/validate_predictions.py --data examples/interface_example/data_dev.json --predictions examples/interface_example/predictions_dev.json --split dev --report logs/validation/interface_example_20261016.json
python -X utf8 src/validate_predictions.py --data data/IMCS-V2-NER.zip --split dev --audit-data --report logs/validation/data_audit_dev_20261016.json
```

第三条可将 `dev` 和报告名分别改成 `train` 或 `test` 核查其他划分。`--data` 支持原始 JSON、数据目录和只读 ZIP。收到真实预测后，用以下模板替换实际路径：

```powershell
python -X utf8 src/validate_predictions.py --data data/IMCS-V2-NER.zip --predictions "<实际运行目录>/predictions_dev.json" --split dev --report "<实际运行目录>/validation_dev.json"
```

`PASS` 返回 0，`FAIL` 返回 1，必需检查无法完成的 `NOT_RUN` 返回 2；参数／输出错误也返回 2。只有空 test 模板时不能核验字符对齐，因此不会通过完整校验。工具不改数据、不修预测、不计算 P/R/F1；run_info 的真实性与完整性仍需人工核查。

## 第二阶段：如何运行统一评分

Python 3.10+、仅标准库、CPU。人工样例全流程：

```powershell
python -X utf8 src/evaluate_predictions.py --data examples/scorer_validation/data_dev.json --predictions examples/scorer_validation/predictions_dev.json --split dev --purpose scorer_validation --output-dir logs/evaluation/manual_cli
python -X utf8 -m unittest discover -s tests -p test_evaluate_predictions.py -v
```

收到真实预测后，假设队友文件存放在 `results/bert_a/run_001/predictions_dev.json`，直接运行：

```powershell
python -X utf8 src/evaluate_predictions.py --data data/IMCS-V2-NER.zip --predictions results/bert_a/run_001/predictions_dev.json --split dev --purpose model_evaluation --output-dir results/bert_a/run_001/evaluation
```

将示例路径换为实际交付位置。支持原始 JSON／目录／ZIP，输入先通过完整校验才评分。每次在 output-dir 下新建唯一子目录；成功写 `evaluation.json`（`status=SCORED`），拒绝时只写不含指标的 `rejection.json`。FAIL／NOT_RUN 和无公开金标准的 test 均不评分。原始数据、预测只读，旧结果不覆盖。

主指标 `entities.micro.f1` 按原始对话／句子内实体类型及 `[start,end)` 严格匹配，全局累计 TP/FP/FN 后计算；五类分别输出。字符指标直接比较各原始 Unicode 码点的 BIO 标签，含 O 的 accuracy、11 类 P/R/F1/support、含／不含 O 的 micro 和固定类别 macro-F1，并保存行=gold、列=predicted 的 11×11 混淆矩阵。排除 O 只排除类别，保留 O 与实体之间的误报／漏报；所有零分母为 0。JSON 保存原始计数、未舍入比例、输入及代码哈希、校验报告和评分耗时。

作者实体段对照：`python -X utf8 tests/crosscheck_author.py --report logs/evaluation/author_crosscheck_20261017.json`。报告独占创建，重复运行换文件名；只执行固定作者源码的实体段，不执行依赖 seqeval 的 token 段及完整 CLI。测试及对照口径详见 [统一评分说明](docs/统一评分说明.md)。

## 第三阶段：一键接入真实预测

先把每次真实预测、run_info、配置及日志放进独立交付目录。复制 [批量清单模板](configs/evaluation_runs.example.json) 为 `configs/evaluation_runs.json`，仅保留已收到的运行并替换实际路径；[运行信息模板](configs/run_info.example.json) 中的待填项必须据实补齐。模板路径不代表真实文件已存在。然后在评估模块根目录执行：

```powershell
python -X utf8 src/run_evaluation_batch.py --manifest configs/evaluation_runs.json --csv-bom
```

相对路径以清单父目录为基准，run_info 内文件以 run_info 父目录为基准。任意数量方法、一个先到也可运行；每次创建 `results/batches/<唯一目录>/`，原始输入和历史结果不覆盖。生成每方法 `evaluation.json`、`reproducibility_check.json`、错误 CSV/JSON，以及总批次报告、整体比较表和五类表。缺少元数据时保留 DEBUG_SCORED 技术评分并明确排除比较；输入校验不通过或 test 无答案则拒绝评分。部分成功返回 2，全部成功 0，失败 1。

`src/export_errors.py` 导出 FP/FN、独立 TP 表、保守一对一差异候选和计数汇总；`src/check_run_info.py` 检查身份、源数据／预测／配置哈希和复现证据；`src/compare_evaluations.py` 复用既有指标汇总可比结果；`src/evaluation_io.py` 提供文件完整性和落盘自检。严格实体和逐字符算法不变，`evaluate_predictions.evaluate()` 新增可选 return_inputs，不改变默认接口和原 CLI。

比较要求同划分、同金标准哈希、同数据版本和指标政策／类别顺序，且输入有效、元数据自动项齐全；人工结果禁止进入真实比较。表中 NEEDS_REVIEW 仍需人工核实模型、无泄漏、映射、复现和计时，不代表实验协议已证明公平。默认无医疗原文，显式 `--local-text` 才导出错误句上下文到 local_analysis；不要纳入公开或课程包。候选差异不改变 TP/FP/FN、不推断根因。

工程测试与完整人工彩排：

```powershell
python -X utf8 -m unittest discover -s tests -p test_validate_predictions.py -v
python -X utf8 -m unittest discover -s tests -p test_evaluate_predictions.py -v
python -X utf8 -m unittest discover -s tests -p test_evaluation_pipeline.py -v
python -X utf8 -m unittest discover -s tests -v
python -X utf8 src/run_evaluation_batch.py --manifest examples/pipeline_validation/evaluation_runs.json
python -X utf8 tests/run_pipeline_acceptance.py --official-data data/IMCS-V2-NER.zip
```

2026-10-17 实跑：原有 42＋47 项及新增 49 项全通过，全量 138 项、0 失败；三方法人工 CLI 为 SUCCESS，最终计数与文件自检 PASS。完整操作与失败定位见[接入指南](docs/真实模型预测接入与评估指南.md)，实跑数量与结果见[第三阶段证据](logs/pipeline_validation/README.md)。人工金标准只有 2 句，三方法总计 6 条预测句子记录，不是正式成绩；官方完整 dev 仅做只读结构兼容性核验。

## 数据与评价约定

已核验的用户下载包位于 `data/IMCS-V2-NER.zip`，本轮不搬入或修改原包。SHA-256：

```text
84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67
```

后续全组固定平台 V2 版本，不混用旧 GitHub 标注。数据来源为[天池 CBLUE](https://tianchi.aliyun.com/dataset/95414)，发布页标注 CC BY-NC-SA 4.0；公开获取可能需要账号申请／学生认证，不能承诺匿名直接下载。

主指标采用严格实体级 P/R/F1，并另行计算真正的逐字符指标。每种方法分别与同一金标准对比；正式实验的数据用途和指标汇总口径须由团队确定并记录。有答案的本地评估用于课程复现；隐藏 test 和天池成绩用于外部验证。保留 dev 作独立评估是前期候选建议，尚未固定具体划分。

第一阶段已按本轮要求固定交接约束：原句字符不得清洗或裁剪，非法 BIO 直接拒绝、不自动修复；完整 dev/test 的统一文件名和元数据见交接规范。模型内部处理须完整还原原句并如实记录。上述约束不决定正式实验划分和指标汇总的实现。

## 后续交付

按真实进度补充依赖与环境、训练／推理／评估命令、种子、预期输出、资源开销、参数对比和失败案例。报告关键数字须能追溯到代码、配置、日志和预测。课程最终只交一个 ≤50MB ZIP，数据、权重、录屏、虚拟环境和密钥不入包。

本项目记录真实 AI 辅助，正式报告另附 AI 使用声明与四份个人贡献声明。当前目录仅是项目起点，不代表课程实现或验收已经完成。
