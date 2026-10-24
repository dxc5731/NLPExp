# IMCS-V2-NER 医疗实体抽取：规则基线 × 大模型对比

> 《自然语言处理》课程项目（2026 秋）
> 一句话：在同一个 train/val 切分、同一把实体级尺子下，对比**非大模型规则基线**与**大模型方案**，并把失败逐一讲清楚。

---

## 0. 30 秒跑通基线（最重要的一条命令）

```powershell
# 1) 进入项目根（本 README 所在目录）
cd <项目根>

# 2) 一条命令得到主结果（纯规则基线，实体级 + 逐字级 P/R/F1）
#    数据放项目根 IMCS-V2-NER/ 即可，程序会自动找到，无需设环境变量
python baseline/run_rule.py 10 min10
```

**预期输出（本机实测，必须一致）**：

```
词表载入：570 词（跨类冲突丢弃 7 个）
...
【逐字级 char-level】
  P=0.8647  R=0.7018  F1=0.7748  acc=0.9421
【实体级 entity-level】
  P=0.7630  R=0.7609  F1=0.7620  (tp=5023 fp=1560 fn=1578)
```

看到 `实体级 ... F1=0.7620` 即复现成功。预测文件写入 `baseline/out/pred_rule_min10.json`，
标准答案写入 `baseline/out/val_gold.json`。

> 若只想要默认词典（min_count=1）的对照结果：`python baseline/run_rule.py`（实体 F1=0.6247）。

> 数据不在项目根时，才需要指定位置：`$env:IMCS_DATA_DIR="...\IMCS-V2-NER"`（自动发现在 `src/config.py`）。

### 0.1 官方 dev 推理与评分（提交用，独立目录不覆盖 val）

```powershell
# 用冻结的 mc10 词典对官方 dev(833 对话/33267 句) 推理，产物写到 baseline/out_dev/
python baseline/run_dev.py
# 用同一把尺子给 dev 预测打分
python baseline/eval_pred.py baseline/out_dev/pred_rule_min10_dev.json --gold baseline/out_dev/dev_gold.json --save
```

**预期（本机实测）**：dev 实体级 `P=0.7564 R=0.7757 F1=0.7659`（即 76.59%）。
val 与 dev 是**两个不同划分**，数字不可混写：val 实体 F1=0.7620、dev 实体 F1=0.7659。

---

## 1. 环境与依赖

### 1.1 跑基线需要什么

| 依赖 | 版本（本机实测） | 必需性 |
|---|---|---|
| Python | 3.12.7 | 必需 |
| numpy | 2.x（实测 2.3.5） | 必需 |
| torch | 1.x/2.x 均可（实测 2.9.1+cpu） | 见下 |

**关于 torch**：基线的算法本身是**纯 numpy 手写**的（TF-IDF、最长匹配都没用 torch，也没用 sklearn）。
但基线读数据复用了 `bert_api/code/src/dataset.py` 里的 `collect_examples`，
而该文件顶部 `import torch`（为 BERT 训练准备），所以**跑基线也连带需要装 torch**。
这不是算法依赖，是复用读取函数带来的连带依赖——如实说明，不做掩饰。

```powershell
pip install numpy torch
```

> **基线段不依赖 `bert_api/code/requirements.txt`**：那份文件是 C 交付 BERT 数据接口时
> 按自己机器 pin 的环境（`torch==2.6.0+cu124` 等），只对"跑 BERT 训练"这一环节有意义。
> 跑基线**不需要**按它装，也不要去降版本——本机能跑通就以本机版本为准。
> 详见 §1.2。

### 1.2 跑 BERT（组内 B/C 负责）额外需要

环境**以 `bert_api/code/requirements.txt` 为准**（C 打包接口时的实测环境）。用法：

```powershell
cd bert_api/code
pip install -r requirements.txt
```

若装 torch 报找不到 `2.6.0+cu124`，改用：
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 有 GPU
# 或没 GPU 时：pip install torch --index-url https://download.pytorch.org/whl/cpu
```

并下载 `bert-base-chinese` 到 `bert_api/code/models/bert-base-chinese/`
（或用 `$env:BERT_MODEL_DIR` 指向它）。

> **两点提醒（跑 BERT 的人注意，不是基线的事）**：
> 1. 上面 pin 的版本我在 PyPI 上都查到了、真实存在（`transformers==5.20.1`、`numpy==2.4.6`、
>    `tokenizers==0.23.3` 均有效），所以能装。
> 2. 但 `torch==2.6.0+cu124` 是 **CUDA 版**，评审/无 GPU 机器上要么装 cpu 版、要么会失败；
>    且它与本机实测（torch 2.9.1+cpu / transformers 4.57.3 / numpy 2.3.5）**不一致**——
>    说明 C 打包时用的是另一台机器。跑 BERT 时以实际能装上、能跑通的版本为准，并在报告中如实记录。

### 1.3 跑 LLM 提示抽取（组内负责 LLM 的成员）

需要 API Key，通过环境变量传入（**绝不写进代码或提交物**）：

```powershell
$env:LLM_API_KEY="sk-..."     # 具体变量名以组内实现为准
```

---

## 2. 数据

- 来源：CBLUE 基准 **IMCS-V2-NER** 任务（中文医疗对话实体抽取），见文末引用。
- 本地目录：`IMCS-V2-NER/`，含 `IMCS-V2_train.json` / `IMCS-V2_dev.json` / `IMCS-V2_test.json`。
- 规模：train 2472 段对话 / 98529 句；dev 833 / 33267；test 811 / 32935。
- 标签：5 类实体（Symptom / Drug / Drug_Category / Medical_Examination / Operation），11 个 BIO 标签。
- **切分**：由 `bert_api/code/artifacts/split_v1.json` 统一规定（`seed=42`，按整段对话 90:10 切 train/val）。
  全组**必须用同一份名单**，否则对比不公平。官方 dev **不参与训练和调参**，只用于最后提交。

**数据放置（自动发现，无需环境变量）**：
把 `IMCS-V2-NER/`（含 `IMCS-V2_train.json`）放到**项目根**（与 `baseline/`、`bert_api/` 同级）即可。
程序在 `bert_api/code/src/config.py` 里按优先级自动查找以下位置，找到就用：
1. `<项目根>/IMCS-V2-NER/`（推荐）
2. `<项目根>/IMCS-V2-NER/IMCS-V2-NER/`（解压多套一层也可）
3. `<bert_api>/doc/数据/IMCS-V2-NER/IMCS-V2-NER/`（原始交接包路径）

数据放在别处时，再设环境变量覆盖：`$env:IMCS_DATA_DIR="...\IMCS-V2-NER"`。

---

## 3. 目录结构

```
自然语言处理作业/
├── README.md                     # 本文件
├── RESULTS.md                    # 所有实验数字 + 出处 + 更正记录（报告直接抄）
├── 项目摘要.md                    # 300 字项目摘要
├── 贡献声明_学号_姓名.md           # 个人贡献声明模板
├── baseline/                     # 非大模型基线（本人负责）
│   ├── run_rule.py               #   纯规则基线·val 跑分（主入口）
│   ├── run_dev.py                #   ★官方 dev 推理（冻结词典，产物进 out_dev/）
│   ├── eval_pred.py              #   ★唯一评测口径（任何方法的 pred.json 都能打分）
│   ├── extract_dict.py           #   从 train 重建词典（产物进 dict_regen/，不碰冻结 dict/）
│   ├── train_tfidf.py            #   TF-IDF 句级 5 类判类
│   ├── gated_combo.py            #   TF-IDF 守门 + 规则组合
│   ├── sweep_mincount.py         #   参数对比：词典 min_count（产物进 sweeps/）
│   ├── sweep_grid.py             #   参数对比：min_count × 阈值 二维（产物进 sweeps/）
│   ├── sweep_tfidf.py            #   参数对比：TF-IDF 阈值（只打印）
│   ├── error_analysis.py         #   错例分析：4 类互斥 + 逐类归因
│   ├── verify_fast_match.py      #   加速实现等价性自检
│   ├── dict/                     #   ★冻结词典（已提交，快速复现直接读这里）
│   ├── out/                      #   val 预测/日志/评测结果（已提交，报告证据）
│   ├── out_dev/                  #   ★官方 dev 预测/标准/评测（独立，不覆盖 val）
│   ├── dict_regen/               #   词典重建产物（可重生成，不入库）
│   └── sweeps/                   #   参数扫描产物（可重生成，不入库）
├── bert_api/                     # BERT 数据接口（C 提供；label_map 等公共依赖来源）
│   └── code/src/                 #   labels.py 为全项目唯一标签来源
└── IMCS-V2-NER/                  # 原始数据（不入库）
```

**输出目录约定（避免互相覆盖）**：
- `dict/` 冻结词典、`out/` val 结果 = **交付证据**，已提交，普通复现只读不写。
- `out_dev/` 官方 dev 结果 = 提交用，独立于 val。
- `dict_regen/`、`sweeps/` = 重建/扫描产物，可随时重跑重生成，**不入库**（见 `.gitignore`）。

---

## 4. 完整实验复现命令

```powershell
cd <项目根>   # 数据放 IMCS-V2-NER/ 于此，自动发现

# —— 快速复现（读冻结词典，秒级；报告主表就用这组）——
python baseline/run_rule.py 10 min10
python baseline/eval_pred.py baseline/out/pred_rule_min10.json --save

# —— 官方 dev 推理 + 统一评分（提交用，独立 out_dev/）——
python baseline/run_dev.py
python baseline/eval_pred.py baseline/out_dev/pred_rule_min10_dev.json --gold baseline/out_dev/dev_gold.json --save

# —— 对照：默认词典 min_count=1 ——
python baseline/run_rule.py
python baseline/eval_pred.py baseline/out/pred_rule.json --save

# —— 词典重建（从 train 重新统计，产物进 dict_regen/，不覆盖冻结 dict/）——
python baseline/extract_dict.py 10

# —— TF-IDF 句级判类 + gated 组合 ——
python baseline/train_tfidf.py
python baseline/sweep_tfidf.py
python baseline/gated_combo.py 0.04
python baseline/eval_pred.py baseline/out/pred_gated.json --save

# —— 参数对比扫描（产物进 sweeps/，可重生成、不入库）——
python baseline/sweep_mincount.py
python baseline/sweep_grid.py

# —— 错例分析（min_count=1 与 10 各一份）——
python baseline/error_analysis.py
python baseline/error_analysis.py 10

# —— 加速实现等价性自检（期望：A不一致=0 B不一致=0 -> 通过）——
python baseline/verify_fast_match.py
```

**接入大模型预测**：把 B/C 产出的预测文件（格式 `{对话ID: {句子ID: "O B-Symptom ..."}}`）
放进 `baseline/out/`，一条命令即可同口径打分：

```powershell
python baseline/eval_pred.py baseline/out/pred_bert.json --tag bert --save
python baseline/eval_pred.py baseline/out/pred_llm.json  --tag llm  --save
```

**日志编码约定**：所有 `out/*.txt` 统一 UTF-8。重定向用
`... 2>&1 | Out-File -Encoding utf8 <log>`，不要用 `*>`（PowerShell 5.1 默认写 UTF-16）。

---

## 5. 评测口径（报告第⑤节照抄）

所有方法**只认 `baseline/eval_pred.py` 这一把尺子**。口径定义在其文件头，两条：

- **逐字级 char-level**：以字符为单位比标签；TP/FP/FN 都是字符数，acc = 完全正确的字符占比。
- **实体级 entity-level**：先把 BIO 还原成 span（起, 止, 类型），三者**全等**才算命中，micro 汇总。

对齐策略：预测里缺的句子按全 O 处理（计入 FN），并报告缺失数量，防止"少交几句反而分高"。

---

## 6. 主要结果（详见 `RESULTS.md`）

| 方法 | 划分 | 逐字 F1 | 实体 P | 实体 R | 实体 F1 |
|---|---|---|---|---|---|
| **纯规则 + 词典 min_count=10**（最佳基线） | val | 0.7748 | 0.7630 | 0.7609 | **0.7620** |
| 纯规则 + 词典 min_count=10 | **dev（提交）** | 0.7843 | 0.7564 | 0.7757 | **0.7659** |
| 纯规则 + 词典 min_count=1 | val | 0.7308 | 0.5295 | 0.7617 | 0.6247 |
| gated 组合（mc=1, θ=0.04） | val | 0.7289 | 0.5885 | 0.7025 | 0.6405 |
| BERT / LLM（组内分工） | val/dev | ⬜ | ⬜ | ⬜ | ⬜ |

关键结论：**词典清洗（min_count 1→10）带来的 +0.1373，是把 TF-IDF 守门员增益（+0.0158）的 8 倍多**；
词典干净后守门员的最优阈值退化为 0（即不起作用）。完整推导见 `RESULTS.md` §3。

> **val 与 dev 不可混写**：val（9687 句，本组自切，用来调参）实体 F1=0.7620；
> dev（33267 句，官方，只用于最后提交）实体 F1=0.7659。两者划分不同、句数不同，报告里分别标注。

---

## 7. 随机性与可复现

| 环节 | 随机性 | 固定方式 |
|---|---|---|
| 数据切分 | 有 | `split_v1.json` 固化，seed=42，全组共用 |
| 规则基线 | 无 | 纯确定性算法，重跑逐字节相同 |
| TF-IDF | 有 | `numpy` 固定 `seed`（见 `train_tfidf.py`） |
| BERT 训练 | 有 | 由 B 负责固定 torch/numpy/random 种子 |

**同一命令重跑，结果应逐位一致**（规则基线的 `pred_rule.json` 已验证重跑逐字节相同）。

---

## 8. 已知限制与诚实说明

1. **规则基线封顶在实体 F1≈0.76**：错例分析（`RESULTS.md` §4.3）显示三类根因——
   无上下文能力（误标）、无冲突消解（边界错）、无长尾覆盖（漏标）——正是规则方法的固有上限。
2. **跑基线需要 torch**：见 §1.1，属复用读取函数的连带依赖，非算法需要。
3. **`bert_api/code/requirements.txt` 的版本与本机不符**：见 §1.2，不影响基线。
4. **词表来自 train 标注**：存在对 train 标注质量的依赖；低频词多为噪声（§3.3 前 8k 词贡献了大部分误标）。
5. **只用 val 调参**：所有超参在 val 上选，dev 未参与。这是本组约定的评测协议（见任务书）。

---

## 9. 分工与 AI 使用

- 分工明细与个人贡献见各组员的 `贡献声明_学号_姓名.md`，与报告第⑧节一致。
- AI 使用声明见报告附录：本项目使用 AI 辅助查文档、调试代码、生成样板、互查错误；
  实验设计、结果分析、结论由组员自行完成并负责。AI 生成内容均经人工核对与修改，日志/数字均来自真实运行。

---

## 10. 数据引用（CBLUE / IMCS-V2）

本项目使用 CBLUE 基准中的 IMCS-V2-NER。若成果中使用了该数据，请引用：

- CBLUE: <https://tianchi.aliyun.com/cblue> · <https://aclanthology.org/2022.acl-long.544/> · <https://github.com/CBLUEbenchmark/CBLUE>
- IMCS（数据来源）：
  - Wei et al., *Task-oriented Dialogue System for Automatic Diagnosis*, ACL 2018.
  - Lin et al., *Enhancing Dialogue Symptom Diagnosis with Global Attention and Symptom Graph*, EMNLP 2019.
  - Chen et al., *A Benchmark for Automatic Medical Consultation System: Frameworks, Tasks and Datasets*, Bioinformatics 2022.

```bibtex
@inproceedings{zhang-etal-2022-cblue,
  title     = "{CBLUE}: A {C}hinese Biomedical Language Understanding Evaluation Benchmark",
  author    = "Zhang, Ningyu and Chen, Mosha and Bi, Zhen and Liang, Xiaozhuan and Li, Lei and Shang, Xin and Yin, Kangping and Tan, Chuanqi and Xu, Jian and Huang, Fei and Si, Luo and Ni, Yuan and Xie, Guotong and Sui, Zhifang and Chang, Baobao and Zong, Hui and Yuan, Zheng and Li, Linfeng and Yan, Jun and Zan, Hongying and Zhang, Kunli and Tang, Buzhou and Chen, Qingcai",
  booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month     = may, year = "2022", pages = "7888--7915",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2022.acl-long.544/"
}
```
