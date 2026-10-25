# NLPExp — IMCS-V2-NER 医疗命名实体识别

《自然语言处理》课程项目，方向 **B：信息抽取 / 序列标注**。

- **数据**：IMCS-V2-NER（CBLUE 医疗基准，公开数据集，[天池入口](https://tianchi.aliyun.com/dataset/95414)）
- **任务**：给定医患对话中的一句话，为每个字符打 BIO 标签，识别 5 类实体
  （症状 / 药物 / 药物类别 / 检查 / 操作）
- **交付**：逐字符 BIO 预测，双层 ID 结构 `对话ID -> 句子ID -> 标签串`

## 仓库结构

本项目由多条各自独立开发的分支合并而成，按模块分目录：

| 目录 | 模块 | 内容 |
|---|---|---|
| `baseline/` | 规则 / 词典基线 | 训练集实体词典、最长匹配、阈值与组合策略、错误分析 |
| `code/` | 公共数据 + BERT-Softmax | 数据切分与子词对齐、Dataset/DataLoader、BERT 微调、dev/test 预测导出 |
| `src/` `configs/` `tests/` `docs/` | BERT-CRF | 字符级 CRF、BIO 硬约束、训练器与阶段验收记录 |
| `evaluation/` | 统一评估 | 预测格式校验、实体级/字符级评分、错误导出、复现证据核查、多模型比较 |
| `doc/` | 文档 | 课程文档、组内分工、接口说明、实验记录、附录材料 |
| `deploy/` | 服务器脚本 | 训练包打包与运行脚本 |

各模块的详细说明：

- 规则基线：[`baseline/`](baseline/)、[`RESULTS.md`](RESULTS.md)
- BERT-Softmax：`doc/组内有用的文档/BERT数据接口说明_v2.md`
- BERT-CRF：[`src/bert_crf/`](src/bert_crf/)、[`docs/bert_crf/README.md`](docs/bert_crf/README.md)
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

## 复现

复现步骤与 2026-10-25 的实测记录见 [`doc/附录A_复现指南.md`](doc/附录A_复现指南.md)，
代码结构说明见 [`doc/附录B_代码结构说明.md`](doc/附录B_代码结构说明.md)。
