# 附录 A：复现指南

> 本附录给出"别人拿到仓库后怎么跑起来"的完整步骤，以及 2026-10-25 实际执行的一次复现记录。

## 0. 环境

| 项 | 本组实测 |
|---|---|
| 操作系统 | Windows 11；Linux（Ubuntu，服务器） |
| Python | 3.11（本地）/ 3.13.5（服务器） |
| PyTorch | 2.6.0+cu124 |
| transformers | 5.20.1 |
| GPU | RTX 4060 Laptop 8G（本地）/ A800-SXM4-40GB（服务器） |

只跑单元测试和评估模块**不需要 GPU**；训练需要 GPU（CPU 也能跑，但慢一个数量级）。

## 1. 准备（数据和模型都不在仓库里）

```powershell
# 数据：从 https://tianchi.aliyun.com/dataset/95414 下载 IMCS-V2-NER.zip，解压后
# 放到 doc/数据/IMCS-V2-NER/IMCS-V2-NER/，或设置：
$env:IMCS_DATA_DIR = 'D:\datasets\IMCS-V2-NER\IMCS-V2-NER'

# 模型：bert-base-chinese 下载到 code/models/bert-base-chinese/，或设置：
$env:BERT_MODEL_DIR = 'D:\models\bert-base-chinese'
```

两者缺任一个，程序会直接说明缺哪个目录、该设哪个变量。

## 2. BERT-Softmax 模块

```bash
cd code
pip install -r requirements.txt                 # torch 按 CUDA 版本另装
python -m unittest discover -s tests -v         # 13 个单元测试

python -m src.train --lr 5e-5 --epochs 3        # 训练（全量约 11 分钟 @ A800）
python -m src.predict --run-id <run_id> --split dev   # 导出预测
```

**预期输出**：测试打印 `OK`；训练每轮打印训练损失、验证损失、逐 token 与实体级指标；
预测打印 `自检通过：833 段对话 / 33267 句`，并在 `runs/<run_id>/predict_dev/` 下生成
`predictions_dev.json` 和 `run_info.json`。

## 3. 评估模块

```bash
cd evaluation
python -X utf8 src/validate_predictions.py --data <数据> --predictions <预测> --split dev --report <报告>
python -X utf8 src/evaluate_predictions.py --data <数据> --predictions <预测> --split dev --purpose model_evaluation
```

**预期输出**：校验器打印 `PASS ... errors=0`；评分器打印实体级 P/R/F1 与字符级准确率。

## 4. BERT-CRF 模块

见 [`docs/bert_crf/README.md`](../docs/bert_crf/README.md) 与
[`docs/bert_crf/08_training_preflight_and_integration.md`](../docs/bert_crf/08_training_preflight_and_integration.md)，
该模块使用独立的 `.venv-bert-crf` 环境。

## 5. 2026-10-25 实际复现记录

从仓库导出干净检出（`git archive HEAD`，文件内容与 `git clone` 一致，不含 `.git`、数据、模型、虚拟环境），当时的检出共 **484 个文件**（此后仓库又补了两个脚本和本附录）。

| 步骤 | 命令 | 实际结果 | 耗时 |
|---|---|---|---|
| 单元测试 | `python -m unittest discover -s tests -q` | **13 passed, 0 failed** | 12 秒 |
| 小规模训练 | `python -m src.train --limit-dialogues 100 --epochs 1` | 实体 F1 0.6839（100 段对话，仅证明流程） | 27.1 秒 |
| dev 预测 | `python -m src.predict --run-id repro_smoke --split dev` | 833 对话 / 33267 句，5901 处非法 BIO 被修复 | 69.3 秒 |
| 官方校验器 | `python src/validate_predictions.py --split dev` | **PASS, errors=0** | 3 秒 |
| 评估模块测试 | `python -m unittest discover -s tests -q` | **135 passed, 0 failed** | 74 秒 |

说明：小规模训练只用于验证流程可跑通，**其 F1 不代表模型效果**；正式结果见报告第⑤节。

## 6. 已知限制

- 干净检出不含数据、模型、权重与 Checkpoint，需要按第 1 步自行准备。
- 所有路径都可用环境变量覆盖，不需要改代码。
- 训练脚本默认 `seed=42`，开启 `cudnn.deterministic`；同机器同环境重跑结果一致。
