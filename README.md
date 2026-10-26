# NLPExp — IMCS-V2-NER 医疗命名实体识别

《自然语言处理》课程项目，方向 **B：信息抽取 / 序列标注**。

- **数据**：IMCS-V2-NER（CBLUE 医疗基准，公开数据集，[天池入口](https://tianchi.aliyun.com/dataset/95414)）
- **任务**：给定医患对话中的一句话，为每个字符打 BIO 标签，识别 5 类实体
  （症状 / 药物 / 药物类别 / 检查 / 操作）
- **交付**：逐字符 BIO 预测，双层 ID 结构 `对话ID -> 句子ID -> 标签串`

## 一条命令复现

**前提**：Python 3.9+，装好 `code/requirements.txt` 里的依赖（torch 需按 CUDA 版本另装）。
数据和模型**不随仓库分发**，按下面第 0 步准备。

```bash
# 0) 准备：下载预训练模型（约 400MB）；数据从天池下载后解压到 doc/数据/IMCS-V2-NER/IMCS-V2-NER/
python reproduce.py --download-model

# 1) 只检查环境和数据（约 5 秒）
python reproduce.py --check

# 2) 快速复现（约 2 分钟）：验证流程与输出是否对得上
python reproduce.py --quick

# 3) 全量复现（A800 约 11 分钟 / RTX 4060 约 71 分钟）：复现主结果
python reproduce.py --full
```

**预期输出**（seed = 42）：

| 模式 | 阶段 | 预期值 | 容差 |
|---|---|---|---|
| quick | 单元测试 | 13 passed | — |
| quick | 验证集实体 F1（100 段对话、1 轮） | 0.7228 | ±0.06 |
| quick | 官方校验器 | PASS，errors = 0 | 精确 |
| full | 验证集实体 F1（全量 88842 句、3 轮） | 0.8885 | ±0.005 |
| full | dev 实体 F1 | 0.8801 | ±0.005 |

**看到什么算成功**：脚本最后打印的汇总表全部 `PASS`，退出码为 0。

**随机种子**：`seed = 42`，同时固定 Python / NumPy / PyTorch 种子并开启 `cudnn.deterministic`。
同机器、同环境下重跑结果一致；不同 GPU 之间可能有 ±0.001 量级差异。

**没有 GPU 也能跑**：`--check` 与 `--quick` 在 CPU 上可用（`--quick` 约 5—10 分钟），
`--full` 建议在 GPU 上跑。

**耗时参考**：见 [`doc/耗时参考.md`](doc/耗时参考.md)。

## 仓库结构

本项目由多条各自独立开发的分支合并而成，按模块分目录：

| 目录 | 模块 | 内容 |
|---|---|---|
| `baseline/` | 规则 / 词典基线 | 训练集实体词典、最长匹配、阈值与组合策略、错误分析 |
| `code/` | 公共数据 + BERT-Softmax | 数据切分与子词对齐、Dataset/DataLoader、BERT 微调、dev/test 预测导出 |
| `src/` `configs/` `tests/` `docs/` | BERT-CRF | 字符级 CRF、BIO 硬约束、训练器与阶段验收记录 |
| `handoff_packages/` | BERT-CRF 结果包 | dev/test 预测、run_info、评分与复现证据（**zip 形式，见目录内 README**） |
| `evaluation/` | 统一评估 | 预测格式校验、实体级/字符级评分、错误导出、复现证据核查、多模型比较 |
| `doc/` | 文档 | 课程文档、组内分工、接口说明、实验记录、附录材料 |
| `deploy/` | 服务器脚本 | 训练包打包与运行脚本 |

## 三种方法的预测产物在哪

| 方法 | 预测文件 | dev 实体 F1 |
|---|---|---:|
| 规则 / 词典基线 | [`baseline/out_dev/pred_rule_min10_dev.json`](baseline/out_dev/pred_rule_min10_dev.json) | 0.7659 |
| BERT-Softmax | [`runs/lr5e-5_ep3_20261022-151229/predict_dev/predictions_dev.json`](runs/lr5e-5_ep3_20261022-151229/predict_dev/predictions_dev.json) | 0.8801 |
| BERT-CRF | [`handoff_packages/`](handoff_packages/README.md) 内 v1.1 压缩包的 `dev/predictions_dev.json` | 0.8935 |

三份预测都在 dev 上、用 `evaluation/` 的同一个评分器计算，口径一致。

各模块的详细说明：

- 规则基线：[`baseline/`](baseline/)、[`RESULTS.md`](RESULTS.md)
- BERT-Softmax：`doc/组内有用的文档/BERT数据接口说明_v2.md`
- BERT-CRF：代码 [`src/bert_crf/`](src/bert_crf/)、阶段记录 [`docs/bert_crf/README.md`](docs/bert_crf/README.md)、**结果包 [`handoff_packages/README.md`](handoff_packages/README.md)**
- 评估：[`evaluation/README.md`](evaluation/README.md)

## 快速开始（规则基线，无需 GPU）

```bash
# 数据放项目根 IMCS-V2-NER/，程序会自动找到
python baseline/run_rule.py 10 min10
```

结果与参数对比见 [`RESULTS.md`](RESULTS.md)。

## 快速开始（BERT-Softmax 模块）

**第 0 步：准备数据和模型（两者都不在仓库里）**

```powershell
# 数据集：从 https://tianchi.aliyun.com/dataset/95414 下载后解压到
#   doc/数据/IMCS-V2-NER/IMCS-V2-NER/    （或设 IMCS_DATA_DIR 指向它）

# 预训练模型：下载 bert-base-chinese 到 code/models/bert-base-chinese/
#   （或设 BERT_MODEL_DIR 指向已下载的模型目录）
$env:BERT_MODEL_DIR = 'D:\models\bert-base-chinese'
$env:IMCS_DATA_DIR  = 'D:\datasets\IMCS-V2-NER\IMCS-V2-NER'
```

没设对的话，程序会直接告诉你缺哪个目录、该设哪个变量，不会抛底层库的奇怪报错。

**第 1 步：装依赖并跑单元测试**

```bash
cd code
pip install -r requirements.txt          # torch 需按 CUDA 版本另装
python -m unittest discover -s tests -v  # 13 个单元测试，约 12 秒
```

**第 2 步：训练与预测**

```bash
python -m src.train --lr 5e-5 --epochs 3             # 训练
python -m src.predict --run-id <run_id> --split dev  # 推理导出
```

## 快速开始（BERT-CRF）

代码在 `src/bert_crf/`，配置在 `configs/bert_crf/`，**结果在 [`handoff_packages/`](handoff_packages/README.md) 的 v1.1 包里**
（dev 实体 F1 **0.8935**，P 0.8757 / R 0.9120）。

权重与原始数据不公开，需要向模块负责人索取（E5 checkpoint 约 1.2GB，SHA-256 见包内 `reproduce/README.md`）。

```bash
# 单元测试（不需要权重）
python -m pytest tests/bert_crf -q

# 启动前预检：检查数据与资源，不训练
python scripts/train_bert_crf.py --config configs/bert_crf/lr_1e-5.json --preflight

# 正式训练：需要已授权的数据与权重，必须显式加 --train
python scripts/train_bert_crf.py --config configs/bert_crf/lr_1e-5.json --train
```

推理复现（生成 dev/test 预测）见包内 `reproduce/README.md`，或直接用 v1.1 包里已冻结的
`dev/predictions_dev.json`、`test/predictions_test.json`。

阶段记录与四轮实验对照见 [`docs/bert_crf/README.md`](docs/bert_crf/README.md)。

## 评估模块

```bash
cd evaluation
python -X utf8 src/validate_predictions.py --data <数据> --predictions <预测> --split dev --report <报告>
python -X utf8 src/evaluate_predictions.py --data <数据> --predictions <预测> --split dev --purpose model_evaluation
python -X utf8 src/check_run_info.py --evaluation <评分结果> --run-info <run_info.json> --run-id <id> --method <方法名> --output-dir <输出>
```

## 约定

- 原始医疗数据、模型权重、Checkpoint、虚拟环境**不入库**，只用链接或路径定位。
- 官方 **dev 不参与训练、调参或模型选择**，只用于最终预测与报告。
- 原始对话 ID 和句子 ID 原样保留，不重新编号；每个原始字符一个标签（含标点与空格）。
- 各模块的 AI 使用范围与程序验证方式见各自文档；正式实验设计、结果分析与个人贡献声明由组员本人完成。

## 训练数据研究与评分核验

[研究成果](evaluation/research/data_study/README.md)包含内部训练集的聚合统计、4 张 PNG/SVG 图、5 份 CSV、研究脚本与 19 项测试，以及八个人工评分场景的手算 / 本项目 / 作者实体评分对照。
[数据来源核查](evaluation/research/data_study/天池与GitHub数据来源核查.md)说明平台 ZIP 与作者仓库的实际版本及标注差异。

## 复现

复现步骤与 2026-10-25 的实测记录见 [`doc/附录A_复现指南.md`](doc/附录A_复现指南.md)，
代码结构说明见 [`doc/附录B_代码结构说明.md`](doc/附录B_代码结构说明.md)。
