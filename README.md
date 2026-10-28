# NLPExp — IMCS-V2-NER 医疗命名实体识别

《自然语言处理》课程项目：对中文医疗对话进行逐字符 BIO 标注，比较规则 mc10、BERT-Softmax、字符级 BERT-CRF。五类实体为 Symptom、Drug、Drug_Category、Medical_Examination、Operation。原始对话/句子 ID、字符、空格及标点保持不变。

## 交付范围与资源

保留完整源码、正式预测、评分、训练与分析证据、演示应用和复现说明。

- 数据：[IMCS-V2-NER 天池入口](https://tianchi.aliyun.com/dataset/95414)。下载并解压后设置 `IMCS_DATA_DIR`；官方 dev 为 833 对话、33,267 句、441,339 字符。
- 冻结双模型：[百度网盘，提取码 uknn](https://pan.baidu.com/s/1-KdQfJJLQTruKY_d5DM3pw?pwd=uknn)。大小、SHA-256、原始验收见 [冻结模型说明](docs/frozen_models/README.md)。微调 checkpoint 与从头训练所需的预训练 BERT 本体是不同资源。
- [演示视频，提取码 qa8w](https://pan.baidu.com/s/1TDMxuzl9h9-0VCLm4Us4FQ?pwd=qa8w)。
- [项目摘要](项目摘要.md)；[交付与打包说明](docs/delivery/README.md)；[新增材料来源和双哈希](docs/delivery/provenance.json)。

## 一条命令与预期输出

**已配置完整本地演示资源**时，在项目根目录用 ML 环境的 Python 执行：

```powershell
python -B -X utf8 scripts/run_dev_demo.py --split dev --methods rule,softmax,crf
```

真实调用三条路线，创建新的 `runs/demo_runs/dev_demo_<时间>_<随机串>/`，生成各路线 `prediction/predictions_dev.json`、`validation.json`、`run_info.json`，以及统一 `evaluation/`、manifest 和日志。完整成功退出 0，部分成功 2，失败/中断 1。它只推理、校验、评分，不训练；配置方法见 [演示使用说明](docs/demo/使用说明.md)。**干净下载的仓库不含本机 runtime 或模型，不能跳过资源准备直接执行该命令。**

新机器优先按 [冻结模型指南](docs/frozen_models/README.md) 使用网盘完整包的相对路径入口进行 Softmax/CRF 推理，无需还原作者机器路径。前端可先以历史浏览模式启动；实时演示的旧适配器仍需配置原模型布局。

无需模型、数据和第三方依赖的一条交付检查命令：

```powershell
python -B -X utf8 scripts/package_delivery.py --check
```

预期输出 `status: PASS` 及检查文件数量，仅核验源码交付清单和压缩包成员；这不是模型运行验收。

## 历史正式结果

以下是**已有真实运行** `dev_demo_20261027_040355_4582ac8c` 的官方 dev 严格实体级评分，非网页预填的本次成绩。本次仓库整理未重新预测、训练或评分。

| 路线 | Precision | Recall | Micro-F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 规则 mc10 | 75.6379% | 77.5723% | 76.5929% | 18,290 | 5,891 | 5,288 |
| BERT-Softmax | 85.3195% | 90.8813% | 88.0127% | 21,428 | 3,687 | 2,150 |
| BERT-CRF E5 | 87.5743% | 91.1994% | 89.3501% | 21,503 | 3,051 | 2,075 |

[日志、校验、评分与预测定位](docs/demo/evidence/README.md)。自动元数据检查 PASS 与人工 `NEEDS_REVIEW` 分开保留；第一阶段旧 run_info 的 FAIL 也未改写。内部 val 用于选模，不能当作官方 dev；test 无公开金标，不提供 test F1。

## 环境与依赖

| 用途 | 环境与安装依据 |
|---|---|
| 交付检查、统一评估 | Python 标准库；`evaluation/` 不要求 GPU |
| 演示后端 | Python 3.12；`python -m pip install -r backend/requirements.lock.txt` |
| 前端 | 已有验收 Node 24 / npm 11；进入 `frontend` 后 `npm.cmd ci` |
| 双模型冻结包 | 历史验收 Python 3.13.5、torch 2.14.1+cu132、CUDA 13.2、transformers 5.20.1、tokenizers 0.23.3、RTX 5060 Laptop；详见包内 requirements 与指南 |
| Softmax 从头训练 | `code/requirements.txt`（服务器环境另见 `code/requirements-server.txt`）；原训练环境与冻结验收环境分开记录，torch 的 CUDA wheel 需用对应官方索引 |
| CRF 从头训练 | `docs/bert_crf/requirements-stage1a.lock.txt` 和 `configs/bert_crf/`，正式训练实际环境见各 run 的 environment.json |

正式两种 BERT 训练随机种子均为 **42**；数据切分种子 42。规则冻结匹配无随机过程。其他 seed 对照仍保留在原实验目录，不能混称正式权重。不同环境结果若有差异，应保留实际预测和说明，不改分数强行对齐。

## 前端与后端

在项目根目录准备后端，在前端目录安装依赖：

```powershell
py -3.12 -m venv .local/demo_backend_env
..localdemo_backend_envScriptspython.exe -m pip install -r backend/requirements.lock.txt
Set-Location frontend
npm.cmd ci
Set-Location ..
```

两个终端分别运行 `.scriptsstart_demo_backend.cmd` 和 `.scriptsstart_demo_frontend.cmd`，访问 http://127.0.0.1:5173 。未配置私有模型也能查看仓库自带的历史成绩和 CRF 四轮分析；资源状态如实显示缺失。完整 dev 只由终端命令触发；网页短文本功能必须配置真实模型，未配置时明确拒绝。未提供已核验官方原句时不显示金标，不上传原医疗文本。[完整使用说明](docs/demo/使用说明.md)。

## 从头训练与独立评估

从头训练需要另备原训练数据与 bert-base-chinese 预训练本体。以下训练命令与冻结推理不同，本轮没有执行：

```powershell
python reproduce.py --check             # 仅检查 Softmax 环境和数据
python reproduce.py --download-model    # 下载预训练模型，需要联网
python reproduce.py --quick             # 会训练：100段对话、1轮
python reproduce.py --full              # 会训练：Softmax全量训练与评估
python scripts/train_bert_crf.py --config configs/bert_crf/lr_1e-5.json --preflight
python scripts/train_bert_crf.py --config configs/bert_crf/lr_1e-5.json --train
```

不带参数的 `reproduce.py` 默认进入 quick 训练；只想检查时必须显式使用 `--check`。

训练配置、耗时口径、原复现记录见 [原附录 A](doc/附录A_复现指南.md)、[耗时参考](doc/耗时参考.md)、[CRF 文档](docs/bert_crf/README.md)。当前 CRF 源码保留第四轮辅助头能力；正式 E5 推理必须用匹配冻结源码，不能拿辅助头权重替代。

任一新预测均可用同一评估器（替换尖括号路径，输出选新目录）：

```powershell
python -X utf8 evaluation/src/validate_predictions.py --data <官方dev.json> --predictions <新预测.json> --split dev --report <新目录/validation.json>
python -X utf8 evaluation/src/evaluate_predictions.py --data <官方dev.json> --predictions <新预测.json> --split dev --purpose model_evaluation --output-dir <新评分目录>
```

仅评分旧预测不能称为模型新推理。指标采用类型与 `[start,end)` 边界完全匹配，字符坐标为原 Unicode 码点。[评估模块说明](evaluation/README.md)。

## 仓库结构与实验材料

| 路径 | 内容 |
|---|---|
| `baseline/` | 冻结五类词表、匹配逻辑、阈值/组合实验及正式结果；[RESULTS.md](RESULTS.md) |
| `code/`、`runs/` | 公共数据接口、Softmax 训练/推理、原配置和记录（不含权重） |
| `src/bert_crf/`、`configs/bert_crf/`、`tests/bert_crf/` | CRF 实现、配置与测试 |
| `docs/bert_crf/training_records/` | 四轮五个 run 的 history/config/metrics、源版本、日志摘录及中断恢复记录 |
| `docs/bert_crf/analysis/`、`docs/bert_crf/figures/` | 原四轮分析和图表 |
| `docs/report_assets/` | 报告实际使用的 8 张 PNG、7 张 SVG、20 Epoch 数据及绘图脚本 |
| `docs/frozen_models/` | 双模型网盘包说明、权重身份与历史验收，不含权重或重复推理源码 |
| `docs/demo/evidence/` | 代表性全量 dev 和 Softmax 冻结验收的公开摘要 |
| `backend/resources/history/` | 可移植历史评分资源，不依赖仓库外审计目录 |
| `frontend/`、`backend/`、`scripts/start_demo_*.cmd` | 本地演示应用 |
| `evaluation/` | 统一评估、数据研究、真实与合成核验记录 |
| `handoff_packages/` | 既有 CRF 正式结果包，保留其版本及来源，不重复上传同一预测 |
| `doc/`、`deploy/` | 已有有效实验文档、接口规范和服务器训练脚本 |

其他成员的已有研究、训练日志、分析及贡献证据继续保留。清理仅移除批准的旧工具和课程模板；历史文档中相关引用作为历史保留，当前入口以本 README 为准。
