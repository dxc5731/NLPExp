# BERT-CRF 阶段 0：真实启动审查

审查日期：2026-10-19，Asia/Shanghai（UTC+08:00）。工作区：`C:/userProgram/NLPexp`。
本记录只覆盖开工审查、只读源码检查及轻量函数探针；没有训练、调参、正式预测、提交或模型成绩。

证据目录：`docs/bert_crf/evidence/20261019T160537+0800/`，下文记为 **E/**。
自动采集从 16:05:37+08:00 开始；此前目录和课程文件阅读来自本次会话的终端输出，未伪造独立历史日志。

状态含义：**实测**＝本机本轮实际执行；**静态**＝阅读源码所得；**交付记录**＝队友包内陈述或 JSON，未在本机复测；**未验证**＝没有足够运行证据；**待确认**＝需要成员作技术选择。

## 1. 材料与项目现状

### 1.1 实际目录及版本

开始检查时根目录只有以下六个文件，没有已解压源码、模型、日志、测试目录或 Git 仓库：

```text
C:/userProgram/NLPexp/
├── 课程项目任务书_20261013_codex.md
├── 项目报告模板_20261013_codex.md
├── 贡献声明表_20261013_codex.md
├── 预测结果交接规范.md
├── IMCS-V2-NER.zip
└── BERT数据接口_v1.zip
```

已完整阅读三个课程文件、预测协议，以及接口 ZIP 内所有 Python、README、依赖清单、使用说明和审计 JSON；名单文件读取其完整 JSON，并核验实际 ID 集合。

| 材料 | 本轮核实 |
|---|---|
| 三个课程文件 | 全部存在；任务书截止时间为 2026-10-30 24:00 |
| 预测结果交接规范 | 第一阶段 v1.0，2026-10-16；已读全部内容 |
| 官方数据 ZIP | 6,369,806 字节；SHA-256 与协议固定值完全一致 |
| 公共接口 ZIP | 24,599 字节；自述 v1；SHA-256 为 `96d259bc1b411a972b46869573f095b0af1de68e2750bdb9ba72450526723df6` |
| 内部划分 | `code/artifacts/split_v1.json`，version=v1，seed=42，val_ratio=0.1，created_by=member_A |
| 原作者源码 | 本轮只下载参考源码及 README，固定 commit `8d512c6635029bf068712d62cf2428e65c5e4a58`；没有下载新数据或权重 |

数据 ZIP SHA-256：

```text
84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67
```

官方 train 原始 JSON 字节 SHA-256（与名单一致）：

```text
07cdd3799087283a5a17de8bb43e71cb159a9a2c99c1cfbe8081847fac358f24
```

六份输入文件及 ZIP 内文件的哈希分别保存于 E/`input_manifest.json`、`dataset_archive_manifest.json`。没有重新序列化数据来代替原文件哈希。

接口实际结构：

```text
README_先看我.md
doc/数据接口说明.md
code/
├── requirements.txt
├── artifacts/{split_v1.json,alignment_report_train.json,
│              alignment_report_dev.json,alignment_report_test.json}
├── scripts/demo_dataloader.py
├── src/{__init__.py,config.py,data_io.py,split.py,labels.py,align.py,dataset.py}
└── tests/test_align.py
```

为了不侵入成员的工程，本轮将接口原样解压到 E/`interface_snapshot/`，没有在根目录铺开 `code/`。数据 ZIP 保持原地只读，没有把医疗原文写入证据日志。

### 1.2 已有成果、缺失材料与 Git

已有：固定划分、数据读取、Tokenizer offsets 对齐、Dataset/动态 padding DataLoader、11 标签表、6 个对齐测试源码、演示入口、三份对齐汇总 JSON。

未交付／未找到：BERT 权重与 Tokenizer 文件、Softmax 模型/训练结果、BERT-CRF 实现、正式实验配置、训练日志/checkpoint/预测、原项目 Git 历史。不能把“未在本工作区找到”解释为队友从未完成。

协议引用的 `src/validate_predictions.py`、`examples/interface_example/`、`logs/validation/data_audit_*.json`、`前期核验/`、`课程要求核对.md`、`本地评估与天池提交说明.md` 均未随本次六个文件或接口 ZIP 交付。因此统一校验器和协议所述原始数据审计 PASS 记录无法在本机执行／查阅；没有重复实现评估成员工具。

`git status --short`、`git log -5 --oneline` 均退出 128：当前目录及父目录不是 Git 仓库。未执行 git init、未补造提交。以输入 ZIP 哈希和逐文件快照哈希定位本轮代码。上游 commit 只是参考代码版本，不是本组提交。

### 1.3 从课程文件提取的硬性要求

- B 方向明确允许 BERT 微调接 CRF；必须和规则/词典及小模型等非大模型基线比较。本人只负责 CRF 模块，不接管公共管线和统一最终评估。
- 必须区分逐 token 指标与严格实体级 P/R/F1；至少一组真实参数对比、至少三类错误归因、至少三个课程原理点且有真实实验数据支撑。概念解释与人工接口探针不算性能实验。
- 模型名/版本、依赖、设备、种子及关键参数要可定位，参数外置；README 给环境、一条可执行命令及预期输出。主结果不能复现有硬性扣分规定。
- 分值含“进度与工作量证明”；报告数字、曲线必须能追溯源码、日志或实际输出，不能生成模拟训练曲线或伪造失败。
- 报告按八节结构，正文 8—12 页；演示录屏不超过 2 分钟，真实运行并含难例，显示学号姓名；每人贡献声明与报告分工一致，报告含 AI 使用声明。
- 最终只交一个不超过 50MB 的 ZIP；数据、模型权重、录屏、虚拟环境、checkpoint 不进课程包，提供可访问链接。保存模型以便复现与课程压缩包不包含模型并不矛盾。
- 课程明确禁止“用 AI 代写实验设计与结果分析”。本记录是技术审查和备选实现说明；正式实验设计、最终分析及贡献声明应由成员独立完成、审核，不把本文当作研究结论。

数据使用遵从用户本轮约束：固定官方 train 内部划分；val 选模；官方 dev 冻结后独立评估；test 暂不使用。协议目录展示未来 test 交接，并不授权本轮或下一轮使用 test。

## 2. 环境情况

| 项目 | 实际检查结果 |
|---|---|
| OS | Windows 11，build 26200；终端 PowerShell |
| 默认 Python | `C:/apps/anaconda/python.exe`，3.13.5，64 位 |
| 另一个已有环境 | `C:/apps/anaconda/envs/mcmc_env/python.exe`，3.10.19 |
| 默认环境训练依赖 | torch、transformers、tokenizers、torchcrf、TorchCRF、seqeval 均未安装；numpy/pytest 存在 |
| mcmc_env | torch、transformers、tokenizers、torchcrf、TorchCRF 均未安装 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB，compute capability 12.0 |
| NVIDIA 驱动 | 595.79；nvidia-smi 显示 CUDA Version 13.2 |
| 框架 CUDA 可用性 | 未验证：没有 torch，不能执行 torch.cuda.is_available() 或实际 CUDA 算子 |
| 本地模型 | 工作区未交付；默认用户 Hugging Face 缓存目录不存在；未全盘扫描，也未下载 |

驱动显示的 CUDA 版本不等于已安装的 Toolkit 或 PyTorch runtime。不能将 nvidia-smi 成功表述为 BERT GPU 训练已可用。

公共 requirements 固定 torch 2.6.0+cu124、transformers 5.20.1、tokenizers 0.23.3、huggingface-hub 2.2.0、seqeval 1.2.2、numpy 2.4.6、tqdm 4.70.1；没有 CRF 依赖。

只读查询 PyPI 后，上述六项非 torch 精确版本均存在，不能猜测它们是“伪造/不存在版本”；记录的 Transformers/tokenizers/hub 依赖约束相容。numpy 2.4.6 要求 Python >=3.11，因此 mcmc_env 的 3.10 无法原样满足此清单。查询官方 cu124 wheel 索引确认存在 torch 2.6.0+cu124 的 Windows cp313 wheel，不能仅凭 Python 3.13 就声称无法安装。

但 RTX 5060 的 sm_120 与该旧 cu124 构建存在架构兼容风险；本轮没有安装或执行 kernel，未证明兼容或具体报错。下一步宜新建隔离环境，选择明确支持该 GPU 的 PyTorch/CUDA 构建，先做设备和前反向冒烟验证，再固定真实依赖。不能照说明中的无版本 `pip install torch` 后仍声称环境完全等同队友清单。

本轮没有安装/卸载依赖、修改全局环境、激活或更改 mcmc_env。8GB 本地 GPU 值得先验证小 batch 的 BERT-base 可行性；实际峰值显存、速度尚未测量，不能保证完整微调一定能放下，也没有证据表明必须 Colab。如本地兼容性或资源不满足，再由成员决定远程 GPU。

## 3. 数据接口审查

源码位置均相对 E/`interface_snapshot/code/`。

### 3.1 划分及原始数据契约

| 划分 | 对话数 | 句数 | 状态 |
|---|---:|---:|---|
| 官方 train | 2472 | 98529 | 本轮由固定名单覆盖及句数核对确认 |
| 内部 train | 2225 | 88842 | 本轮核验，字符数 1153312 |
| 内部 val | 247 | 9687 | 本轮核验，字符数 126049 |
| 官方 dev | 833 | 33267 | 协议/交付记录；本轮确认整包哈希相同，不重跑全量审计 |
| 官方 test | 811 | 32935 | 协议/交付记录；同上 |

内部名单两边各自无重复、互不相交、并集恰为官方 train 的所有 ID；源文件 SHA-256 一致。没有调用会重写名单的 `python -m src.split`。

官方 train 的本轮轻量契约核验：长度不匹配 0、未知标签 0、严格 BIO 非法转移 0、同一对话内重复 sentence_id 0、非字符串 sentence_id 0、空句 0、含空白字符句子 0、含非 BMP 字符句子 0；内部 train 最大 199 字符，val 最大 159 字符。这里只证明当前 train，没有证明一般 Unicode 输入都能对齐。

实际字段为 `self_report`，句子有 `sentence/sentence_id/BIO_label/speaker` 等，和协议说明吻合。`mappings.json` 的 BIO 子表确实只有 PAD/O/B-Symptom/I-Symptom，不能当五类 NER 标签表。只使用接口的 `src/labels.py`，没有从 dev/test 抽取标签或资源。

### 3.2 真实入口、路径与输出

原说明要求在 **解压根/code** 下执行：

```text
python -m scripts.demo_dataloader
```

`config.py:12` 按文件位置计算解压根；默认数据目录是 `<解压根>/doc/数据/IMCS-V2-NER/IMCS-V2-NER`，默认模型目录 `<解压根>/code/models/bert-base-chinese`。可用 `IMCS_DATA_DIR`、`BERT_MODEL_DIR` 覆盖，必须在 import 之前设置。本次审查快照默认这两个目录均无所需资源。`load_split` 只接受解压目录下的 JSON，不能直接加载当前 ZIP。说明中的 `set NAME=...` 是 CMD 写法，PowerShell 应使用 `$env:NAME=...` 或 Python subprocess 的 env 参数。

`load_tokenizer` 从本地路径 `AutoTokenizer.from_pretrained(..., use_fast=True)` 加载；本轮演示明确设置离线环境变量以避免网络访问。正常调用为：读取已有 split_v1.json 的 train 或 val ID，`load_split('train')`，`collect_examples(data, ids)`，再 `build_loader(examples, tokenizer, ...)`。内部 val 仍取官方 train，不能改为 `load_split('dev')`。

| batch 字段 | 源码约定形状/类型 | 含义 |
|---|---|---|
| input_ids | LongTensor [B,L] | 动态 padding 的 WordPiece ID |
| attention_mask | LongTensor [B,L] | CLS/SEP 与真实词片为 1，padding 为 0 |
| labels | LongTensor [B,L]，有标签时存在 | 0—10；CLS/SEP/零宽 offset/PAD 为 -100 |
| offsets | LongTensor [B,L,2] | 原句 Unicode 字符的半开区间；特殊/补位为 (0,0) |
| dialogue_id / sentence_id / text | 长度 B 的列表 | 原 ID、原句，未拼 speaker，未归一化 |

以上是**静态接口形状**，不是本机打印出的真实 batch。`L` 是 batch 最大 token 长度，默认 max_len=256；演示 batch_size=4，最多取名单前 200 组的句子，只核对第一句往返，不是全量验证。`一致: True` 即使出现，也不能证明全部数据满足 CRF/交接要求。

无标签推理需同时 `collect_examples(..., require_labels=False)` 和 `build_loader(..., with_labels=False)`。推理有效位置要从 attention_mask 和 offsets 构造，不能依赖推理时不存在的 gold labels。BERT 单句输入可以默认 token_type_ids 全 0，接口未提供此字段不构成确定错误。

### 3.3 对齐、还原与测试的具体问题

| 位置 | 发现及验证等级 | 影响 |
|---|---|---|
| align.py:45 | **静态+纯函数实测**：每个非零宽 offset 直接取 char_tags[start]；没有跳过所有 `##` 延续片 | 文件头“续段填 -100”的说法与代码不符，需按真实 offsets 判断 |
| align.py:45 | **实测人工探针**：gold=[O,B-Drug]，一个 span=(0,2)；往返变为 [O,O] | 单个词片跨字符边界时，字符标签不可逆；不是调整 Mask 就能修复 |
| align.py:56 | **静态+实测**：多字符词片复制标签，B 后续转为 I | 仅适合词片内部没有真实边界变化的情况 |
| align.py:71 | **静态+实测**：未覆盖字符直接填 O，并返回 uncovered | 本函数不是 fail-fast；调用者忽略 uncovered 就会静默掩盖缺字，禁止直接用于正式导出 |
| align.py:75 | **静态+实测**：非法 I 自动改成 B | 可能把非法 CRF 输出修成校验能接受的预测，不能据此宣称解码满足 BIO 约束 |
| align.py:59 | **静态+实测**：zip 标签/offset，不校验等长；未校验范围/重叠/标签合法性 | 人工探针中标签少一项未报错，缺失位置被补 O；非零 span 传 -100 会 IndexError |
| align.py:33 | **静态**：truncation=True，无 overflow、完整覆盖断言或还原机制 | 不能保证未来长句尾部保留；必须检查实际 token 长度及每字符覆盖，超长要明确报错或完整分段 |
| dataset.py:121 | **静态**：collate_fn 为 lambda | Windows num_workers>0 有 spawn/pickle 风险；默认 0，可先保持，不改公共代码 |
| tests/test_align.py | **静态**：六个测试；还把“非法 I 被修复”作为预期 | 测试不覆盖严格失败策略、CRF Mask/约束、复杂 Unicode、长句完整还原和全部真实往返 |

人工探针调用的是原始函数，没有替换 torch、伪造 Tokenizer 或模拟模型。完整输入意图和输出在 E/`pure_alignment_probes.json`；这些是接口行为证据，不是 NER 失败案例或模型实验结果。

包内 train 对齐报告写有 98529 句、max_tokens=201、截断/缺字/重叠为 0、金标往返不一致 8 句；dev 报告为 7 句不一致。本轮只是阅读交付 JSON，没有重跑 tokenizer，也没有据 dev 标签选择策略。报告没有关联的数据/Tokenizer 哈希、生成脚本、实际命令、时间戳或逐句诊断，所以这些只能标记为**交付记录**。test 没有金标，其报告的 roundtrip_mismatch=0 不能解读为金标往返全通过。

真实运行结果：

| 命令（工作目录 E/interface_snapshot/code） | 本轮结果 |
|---|---|
| `python -X utf8 -m scripts.demo_dataloader` | exit 1；dataset.py 导入 torch 时 ModuleNotFoundError；未得到 batch，也未得到“一致: True” |
| `python -X utf8 -m unittest discover -s tests -v` | exit 1；测试模块导入 torch 失败，loader 显示 1 个导入错误；六项测试内容均未执行 |

说明里的“6 个全过”没有随包原始测试日志支撑，本机不能复核。不能把上述导入失败解释成六个断言都失败，也不能称测试通过。

### 3.4 CRF 兼容性判断

**可以复用名单、原始数据读取、标签表、原 ID 与 offsets，以及 BERT 输入 batch；不能把当前 labels/attention_mask 直接交给普通 CRF。**

1. 0=O，1/2=Symptom 的 B/I，3/4=Drug，5/6=Drug_Category，7/8=Medical_Examination，9/10=Operation。PAD 不是输出类别，CRF num_tags=11。
2. attention_mask 包含 CLS/SEP；labels=-100 是交叉熵语义，CRF 通常会先索引标签分数，不能假定 Mask 能保护 -100 不被索引。
3. 直接 `labels != -100` 的第一列在 CLS 处是 False，不满足许多 CRF “第一步必须有效”的条件；内部若有忽略位置也不是右 padding 的连续前缀。
4. 可在 CRF 模块局部按原顺序 gather 有效 emissions/labels，压紧后右 padding，构造 Bool mask：每行 True 连续前缀，首步 True；所有张量标签值（包括补位）均为合法 0—10；另保留逆映射供解码还原。空有效序列明确处理，不能硬塞虚构标签。这是对 CRF 输入的适配，不需要重建公共 DataLoader。
5. 该 token 级方案只能处理 Mask，不能恢复已经丢掉的词片内部边界。若要求逐字符无损，应考虑由 offsets 将 BERT hidden states 映射到原字符序列，再按原始字符金标计算 CRF；多字符 span 的表示复制/聚合规则须明确。缺覆盖、重叠、越界必须报错或进入显式处理分支，禁止补 O。原始字符 gold 需从允许的 train/val 样本传给模型适配层；公共 batch 当前只含 token labels，不能从它逆推完整 gold。
6. 特殊 Unicode 可能被清理、归一化或变为 UNK；原文保持不变并不自动保证 offsets 无损。当前 train 无空白/非 BMP，不能代表协议人工 Unicode 示例。应使用批准的真实 Tokenizer 做人工 Unicode 覆盖检查；若必须改变 tokenization，先与 BERT-Softmax 负责人统一，不能单方修改公平对比条件。
7. 普通 CRF 无内置 BIO 保障。句首禁止 I-X，转入 I-X 只允许同类 B-X/I-X；应在训练所用的分区函数和 Viterbi 中持续应用固定约束，而非只初始化为很小数或导出时改标签。约束实现和范围需记录，后续消融由成员设计决定。

## 4. BERT-CRF 技术建议与原作者审查

### 4.1 固定参考版本与可复用思路

参考来源：`https://github.com/lemuria-wchen/imcs21-cblue/tree/8d512c6635029bf068712d62cf2428e65c5e4a58/task1/BERT-NER`。
完整参考文件、下载时间及 SHA-256 在 E/`upstream_manifest.json`；源码在 E/`upstream_snapshot/`。以下均为**静态审查**，没有运行原作者旧环境。

值得复用：BERT 逐位置隐状态 → Dropout → Linear emissions → 线性链 CRF；负对数似然；Viterbi 解码；随机种子、梯度裁剪、warmup 与保存加载的基本流程。不是复用旧结果表中的 F1 作为本组成绩。

| 参考文件与行号 | 真实行为及风险 |
|---|---|
| modeling_nerbert.py:2 | imports transformers.modeling_bert 和已过时 archive map；不能照搬到新 Transformers |
| requirements.txt | 固定 torch 1.4.0、transformers 2.7.0、pytorch-crf 0.7.2、pandas 1.0.3；与本机现代 Python/公共接口不是同一环境 |
| modeling_nerbert.py:49 | CRF 以 attention_mask 为 mask，特殊 Token 的标签也参与序列；默认 ignore_index=0 在原代码代表 PAD，不是公共接口的 -100；两者不可混用 |
| trainer.py:170；predict.py:165 | decode 没有传 Mask；PAD/特殊 Token 可影响真实位置的最优路径，事后过滤不能消除该影响 |
| modeling_nerbert.py:34 | 普通 pytorch-crf，无显式 BIO start/transition 约束 |
| preprocess.py:16；predict.py:53,198 | 拼 speaker+全角冒号、人工加 3 个 O、最后剥 3 标签；不符合本组原字符输入约定 |
| preprocess.py:34,87 | 从 train 和 dev 标签共同构建词表；本组不能复用保留集标签构建资源，使用固定 11 标签即可 |
| preprocess.py:63 | 相对路径 V2/train_update.json、V2/dev_update.json，与当前官方 ZIP 文件名不同 |
| data_loader.py:75,83 | strip/split 文本行，不能忠实保留原文空白；features 也未保留原始对话/句子坐标 |
| data_loader.py:130 | 按“字符词”分词，首片标签、后续片 PAD；与公共接口“整句分词后取 span 起点标签”策略不同 |
| data_loader.py:138；main.py:37 | 直接截断训练序列，默认 max_seq_len=50，不能沿用为当前完整字符方案 |
| trainer.py:47 | 自动有 GPU 就用 cuda:1；本机只有 GPU 0，配置不适用 |
| trainer.py:8,70 | transformers.AdamW 的旧调用，应采用已验证现代环境中的 torch.optim.AdamW |
| trainer.py:118,121,219 | 固定步数验证/保存，保存到同一 model_dir，没有按内部 val 最佳严格实体 F1 留存最佳模型的明确逻辑，也未保存 Tokenizer |
| utils.py:79 | seqeval 默认调用，未显式指定严格 IOB2 模式；不能直接宣称与本组严格指标一致，也没有逐 token 指标 |
| inference.py:49,60 | strip 原文，且预先裁为 510 字符；不能承担完整字符还原 |
| data_loader.py:230；predict.py:23 | torch.load 旧式对象缓存/argparse 参数持久化，在现代默认安全加载行为下有兼容风险；需明确格式及重载验收 |

main.py 的学习率、batch_size、epoch 等大部分是 argparse 默认值，可覆盖，并非全部“硬编码不可改”；但路径映射、GPU 和旧预处理确有固定假设。不要笼统否定全部参考代码，也不要原样运行它的 dev 选模流程。

### 4.2 待确认的实现选择

- 基础模型建议先与成员 B 共用同一份 `bert-base-chinese` checkpoint 和 Tokenizer，固定实际来源、revision/文件哈希与配置，避免模型版本成为混杂因素。本轮没有下载，也未确定最终模型来源。
- CRF 建议采用可审查的线性链实现：可以以 `pytorch-crf` 为基础做显式约束封装，或独立最小实现；`pytorch-crf`（import torchcrf）与 `TorchCRF` 不是同一库。普通现成层不能直接满足本协议，最终选型要经单元验算和现代环境兼容验证。
- 首要决策是 token CRF（继承已知边界投影损失）还是 offsets 映射后的字符 CRF（需要原字符 gold 的局部适配）。从交接的逐字符要求看，字符 CRF 值得优先讨论；不能悄悄让 CRF 与 Softmax 使用不同输入协议。
- 长句可以先明确“超长即失败”，在当前数据与真实 Tokenizer 验证通过后使用；若需要分段，须完整恢复句子坐标、记录边界策略，不能静默裁尾。
- 正式训练前要与评估负责人确认内部 val 的严格实体指标接口、平均方式、逐 token 口径、零分母规则和最佳 checkpoint 判据。本人不另起全组最终评分程序。

## 5. 证据、执行命令与变更

### 5.1 本轮实际执行

1. `Get-Location`、`rg --files --hidden ...`、Python pathlib/zipfile/json 只读材料读取：成功；确认起始只有六份输入、完整读取课程与接口代码。
2. `where.exe python`、`conda env list`、`git --version`、`python -m pip --version`：成功；两套解释器路径如上。
3. `git status --short`、`git log -5 --oneline`：失败（128），如实无仓库。
4. `nvidia-smi` 及精确 GPU CSV 查询：成功；未执行训练算子。
5. `python -X utf8 docs/bert_crf/audit_stage0.py`：退出 0，生成新时间戳目录；其内部演示/测试失败已分别保存，collector 退出 0 不代表接口通过。
6. 上述 collector 中 `python -m pip list --format=json`：成功，保存默认环境实际完整包列表；两条接口运行失败见第 3 节。
7. GitHub API 获取 main 对应 commit、tree，再按 commit 下载 BERT-NER 文本源码与两层 README：成功；未执行参考训练脚本。
8. `python -X utf8 docs/bert_crf/audit_followup.py docs/bert_crf/evidence/20261019T160537+0800`：退出 0，保存第二环境、只读依赖查询、train 契约及源文件完整性记录。
9. 曾尝试一条内联 Python 访问 wheel 索引命令，被 PowerShell 引号解析报 ParserError，Python 未运行；随后改入 Python 文件。
10. `python -X utf8 docs/bert_crf/audit_followup.py docs/bert_crf/evidence/20261019T160537+0800 --wheel-index`：退出 0，成功保存官方索引核验。

早期探索命令的输出存在于本次会话，未全部逐条另存本地；核心版本、环境、命令返回码、traceback 和源文件哈希均已归档。那次 shell 引号失败只有原因/退出码摘要，无独立秒级时间记录，见 E/`shell_failure_note.json`。不把它伪装成有完整逐秒记录。

### 5.2 可定位证据

| 文件/目录（相对 E/） | 内容 |
|---|---|
| input_manifest.json | 六个原始输入的字节数与哈希 |
| interface_manifest.json + interface_snapshot/ | 逐文件原样快照及哈希 |
| dataset_archive_manifest.json | 原包内部各文件原始字节哈希，无数据原文日志 |
| split_verification.json | 固定名单、源哈希、ID 集合、内部样本计数与长度核验 |
| train_contract_checks.json | 仅官方 train 的 BIO/ID/字段/Unicode 范围检查 |
| environment.json + secondary_environment.json | 两套解释器与库可见性 |
| commands/00.json—05.json + commands_index.json | cwd、实际 argv、开始时间、耗时、stdout、stderr、退出码；03 中含 pip 包清单 |
| pure_alignment_probes.json | 人工输入调用既有纯函数的真实行为；无模型运行 |
| dependency_index_checks.json | PyPI 精确版本、Python 要求、依赖与 wheel 文件名；无安装 |
| torch_wheel_index_check.json | 官方 cu124 索引核验，无 wheel 下载 |
| upstream_manifest.json + upstream_snapshot/ | 固定上游 commit、源码、下载时间、文件哈希 |
| source_integrity_check.json | 六份输入及接口快照逐文件哈希全部保持不变 |

仅新增 `docs/bert_crf/`：本报告、两份标准库审查脚本及时间戳证据目录；没有修改原有六份文件或公共源码、数据名单。运行导入曾在新快照内生成少量 Python 字节码缓存，收尾删除自身缓存；它们不是原包内容。本轮脚本最终状态和证据文件的哈希另见 `stage0_final_manifest.json`。

源码快照和哈希已能定位本轮输入，但训练复现尚未建立：没有实际 Tokenizer/权重版本、训练环境、模型测试或日志。后续正式运行应另建唯一 run_id 目录，不覆盖此阶段 0 证据；记录实际起止时间、代码快照/commit、划分哈希、模型/Tokenizer/依赖、全部种子与配置、命令、设备、耗时、训练 Loss、内部 val P/R/F1、最佳 checkpoint 判据、真实预测及异常修复。不要提前创建装作已运行的训练记录。

重跑采集可执行 `python -X utf8 docs/bert_crf/audit_stage0.py`，每次生成新目录，不覆盖旧记录；随后将新目录传给 audit_followup.py。followup 对现有同名证据采用拒绝覆盖。上游 main 可能移动，复核本轮参考代码应优先使用已固定 commit 的快照。

## 6. 下一轮建议与开工条件

**已经满足：**课程/交接要求已读；数据原包身份已核实；固定名单完整可用；标签表和 ID/offset 接口可定位；CRF 直接接入的错误风险已有真实证据；原作者代码已固定版本；阶段 0 独立记录已建立。

**尚未满足：**隔离训练环境；可加载的准确 BERT/Tokenizer 文件；真实 tokenizer 覆盖与金标边界诊断；DataLoader 演示和六项测试本机通过；严格还原策略；CRF 连续 Mask/约束实现；已有统一校验工具和原始审计日志的交付；与内部 val 指标接口对齐。

推荐下一轮最小可验收任务（须收到下一轮指令才执行）：确定隔离环境及模型来源，先跑真实接口演示；在本人模块内完成 BERT → emissions → CRF 的最小适配；用人工小序列验算 NLL/Viterbi/BIO 约束、非连续忽略位置压紧、特殊 Token/PAD/Unicode/长度失败处理，并完成单 batch 前向/反向与保存重载一致性。它不等同首次完整微调，不能预设性能。

需要成员确认：

1. 本地隔离环境还是其他 GPU；允许获取哪份 BERT 权重/Tokenizer，以及公共代码正式放置位置。
2. token 级或字符级 CRF；如何与成员 B 协调对齐修正、已知 8 条 train 往返差异及公平比较。
3. CRF 依赖/实现选型，以及 BIO 约束在训练和解码中的范围。
4. 评估成员何时提供现有校验器、数据审计日志、内部 val 指标契约；不要求为模型结构冒烟先生成官方 dev 预测。
5. 正式训练配置、最佳 checkpoint 判据和第一组受控对比由成员后续确定；本轮不代拟完整实验设计。

阶段 0 到此结束，等待下一条指令，不进入阶段 1。
