# BERT-CRF 阶段 1A：环境与公共数据接口实测

日期：2026-10-19，Asia/Shanghai。运行 ID：`20261019T162653+0800_stage1a`。
工作区：`C:/userProgram/NLPexp`。阶段 0 材料和原证据已复核，保持原样。
本轮完成环境安装、真实 CUDA 计算、公共演示/测试、完整内部 train/val 审计；没有模型训练、CRF 实现、超参搜索或官方 dev/test 预测。

## 1. 实测结论

- **环境验证通过：**Python 3.13.5、PyTorch 2.14.1+cu132、CUDA runtime 13.2、Transformers 5.20.1、tokenizers 0.23.3。RTX 5060 Laptop 的 sm_120 在构建支持列表中，张量核函数、矩阵乘法与反向传播实测通过。
- **接口能运行：**演示显示“一致: True”；随包 6 项 unittest 全部通过；完整内部 train/val 均可构造 Dataset 并获取真实 batch。
- **接口尚不能视为无损：**98529 句、1279361 字符均完整覆盖，但 8 句、11 字符金标往返不同；其中 2 句 token 金标出现非法 O→I。它们会影响 Softmax 和 CRF，属于公共对齐问题。
- **当前数据没有截断：**最大 201 Token（含 CLS/SEP），上限 256。一般空白/Unicode、超长输入的严格处理仍有缺口，已用人工文本和真实 Tokenizer 复现。
- **CRF 适配可独立完成：**实际数据没有内部忽略空洞，但原 Mask 首列是 CLS 的 False，不能直接给普通 CRF；有效位置压紧探针通过，没有实现 CRF 模型。

这不是模型效果实验，不提供或推断 F1。

## 2. 环境安装与兼容性

环境位于 `.venv-bert-crf/`，由现有 `C:/apps/anaconda/python.exe`（3.13.5）创建 venv，`include-system-site-packages=false`。所有 pip 安装都显式使用该环境解释器，没有向 base 或 mcmc_env 安装包。

选择依据是实际查询 PyTorch 官方安装页面及稳定 wheel 索引，而非只看 nvidia-smi。查询时官方稳定版本为 2.14.1，官方 cu132 索引有 Windows cp313 wheel；最终实际安装来源与 wheel 哈希保存在 `installation_sources.json`。驱动支持上限和 wheel 自带 runtime 是不同概念，此次恰好均显示 13.2。

| 项目 | 实际值 |
|---|---|
| Python | 3.13.5，64 位 |
| torch | 2.14.1+cu132 |
| torch.version.cuda | 13.2 |
| torch.cuda.is_available() | True |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Compute Capability | 12.0 |
| 构建支持 | sm_75/80/86/90/100/120 |
| transformers / tokenizers | 5.20.1 / 0.23.3 |
| huggingface-hub / numpy | 2.2.0 / 2.4.6 |
| seqeval / pytest | 1.2.2 / 9.1.1 |
| tzdata | 2026.5 |
| pip check | exit 0，No broken requirements found |

除 torch 外复用公共接口固定的依赖版本，额外安装 pytest、tzdata；CRF 库本轮未安装，不影响本阶段验收。完整实际版本见 `requirements-stage1a.lock.txt`。

唯一失败是首次运行自检时缺少 Windows zoneinfo 所需的 tzdata，抛出 ZoneInfoNotFoundError，发生在 GPU 计算之前。补装 tzdata 后重试通过；两次原始日志均保留。没有发生 GPU 架构报错或反复更换 PyTorch。

CUDA 程序 `gpu_smoke.py` 使用 seed=1729：真实执行 arange/加乘求和及 [128,256]@[256,64] 矩阵运算，CPU/CUDA 分别反向传播并比较结果。CUDA loss=255.10491943359375，CPU loss=255.1049041748047，x/w 梯度最大绝对差约 5.96e-8/1.04e-7；断言通过。这些是数值自检数据，不是模型训练 Loss。

已具备继续开发模型的 Python/GPU 基础；未下载 BERT 权重，未验证 BERT 模型前反向、CRF 库或完整训练峰值显存，不能将当前自检外推为完整微调已通过。暂无必须迁移 Colab 的证据。

## 3. 部署、Tokenizer 和真实 Batch

独立工作目录：`.local/bert_crf/20261019T162653+0800_stage1a/`，其中：

```text
interface/     原 ZIP 原样解压，含 code/src 等
data/          只解压 IMCS-V2_train.json，未解压 dev/test
tokenizer/     公开模型的 4 个 Tokenizer/配置文件，无权重
logs/          原始命令输出，含 demo 打印的医疗文本；不进入 Git
alignment_issues_private.jsonl  8 条详细问题样例，不进入 Git
install_*_report.json          pip 原始安装来源记录
```

通过 subprocess 环境变量设置 `IMCS_DATA_DIR`、`BERT_MODEL_DIR`；同时设置 UTF-8 和离线开关。公共代码无需修改，也没有因审查目录的嵌套路径而错误判定加载逻辑失败。

Tokenizer 来源：`google-bert/bert-base-chinese`，revision=`8f23c25b06e129b6c986331a13d8d025a92cf0ea`。只下载 config.json、tokenizer.json、tokenizer_config.json、vocab.txt；URL、字节数和 SHA-256 见 `tokenizer_manifest.json`。真实加载类型名为 BertTokenizer，`is_fast=True`，底层 BertNormalizer 为 clean_text=True、handle_chinese_chars=True、lowercase=False。不能仅凭类名没有 Fast 后缀认定不是 Fast Tokenizer。

数据及接口 ZIP SHA-256 分别为：

```text
84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67
96d259bc1b411a972b46869573f095b0af1de68e2750bdb9ba72450526723df6
```

原始 train 字节哈希与 split_v1.json 的 source_sha256 一致：`07cdd3799087283a5a17de8bb43e71cb159a9a2c99c1cfbe8081847fac358f24`。内部 2225/247 组名单直接读取，未运行划分脚本。内部 val 也是读取官方 train 后按名单取样，未加载官方 dev。

公共演示实际读取名单前 200 组、7476 句，首 batch 为 4 条，最终打印“一致: True 未覆盖字符: []”。原始演示日志保留本地；没有把原文复制到本报告。

| 字段 | train 首 batch | val 首 batch | dtype/格式 |
|---|---|---|---|
| input_ids | [4,11] | [4,40] | torch.int64 |
| attention_mask | [4,11] | [4,40] | torch.int64 |
| offsets | [4,11,2] | [4,40,2] | torch.int64 |
| labels | [4,11] | [4,40] | torch.int64 |
| dialogue_id/sentence_id/text | 各 4 项 | 各 4 项 | list，原 ID/原句 |

train 首 batch 的 attention 长度为 [4,10,4,11]，标签有效长度 [2,8,2,9]；23 个 -100 由 15 个 padding + 8 个 CLS/SEP 构成。val 首 batch 对应 [40,7,25,15] 与 [38,5,23,13]，81 个 -100=73 padding+8 CLS/SEP。ID 和原文与输入样本一致。

还实际验证了 `require_labels=False` 配合 `with_labels=False` 的无标签加载：batch 不包含 labels，能从 attention_mask 与 offsets 构造有效位置。没有生成任何模型预测。

随包全部六项测试均通过，包含 offsets 覆盖、特殊项忽略、多字符展开、非法 I 自动修复、普通句子往返和 padding。测试通过只说明这些既有断言成立；其中一项恰好把自动修复视为预期，不足以证明严格交接无损。

## 4. 真实对齐质量与 CRF 条件

完整统计、8 条定位和人工边界检查见 [alignment_summary.md](alignment_summary.md)。主要结果：

| 指标 | 实测 |
|---|---:|
| 全量检查句子／字符 | 98529／1279361 |
| 每字符恰好覆盖一次的句子 | 98529 |
| 金标往返一致／不一致句子 | 98521／8 |
| 不一致字符 | 11 |
| 多字符 Token／其中有损 Token | 3825／8 |
| 自动补 O，真实数据 | 0 |
| 自动 I→B，真实金标往返路径 | 2 次，2 句 |
| 实际截断句子／字符／受影响实体 | 0／0／0 |
| 原字符 gold 非法 BIO | 0 |
| 投影后 Token gold 非法 BIO | 2 次 O→I |

8 条中 train 7 条、val 1 条，7 个 Token 跨实体起点，1 个跨实体终点。原始金标结构合法；问题由一个 Token 只能取首字符标签而丢失边界引起。没有修改金标，也没有把“只有 8 条”当作无问题。旧报告不含 ID，能复现总数量，不能证明逐条就是历史同一批。

尤其需要注意：2 条合法字符序列投影成非法 Token gold。如果后续 Token CRF 在训练时硬性禁止 O→I，这些监督路径会被排除；不能简单改 I 为 B 来掩盖投影问题。

当前全量数据没有有效序列中间的忽略空洞；333 个 `##` 续片全部有标签，和文件头“续片填 -100”的陈述不同。CLS/SEP 的 197058 个标签均为 -100，PAD 在真实 batch 中也正确忽略。这些结果纠正了阶段 0 尚未实测的假设。

train/val 的原始标签有效 Mask 首列均为 False，因为 CLS 被排除；attention_mask 则含 CLS/SEP。两份真实 batch 的局部压紧探针已验证：按 offsets 选位置、保留顺序、右 padding、合法标签填充值、Bool Mask 连续前缀及首步有效；原选中标签可精确取回。这里只检查张量适配，没有调用 CRF likelihood 或 Viterbi。

对正式解码仍须在本人模块显式保证 BIO start/transition 约束，并严格检查字符完整覆盖；不能用公共还原函数的自动 I→B 证明解码本身合法。

## 5. 是否需要公共接口修复

| 问题 | 是否真实复现 | 严重性 | 公共接口需要修改？ | 处理人 |
|---|---|---|---|---|
| 字符与 Token 对齐 | 是，8 句／11 字符；覆盖本身全通过 | 正确性问题，影响两种模型；规模小不代表无损 | 是，需共同确定金标/表示对齐方案 | 维护成员 A，协调 B；本人提供复现 |
| BIO 标签修改 | 是，真实往返有 2 次 I→B；真实补 O 为 0，人工缺字样例触发补 O | 自动修复掩盖投影错误和非法输出 | 是，需严格模式/显式失败及调用契约 | A；本人避免依赖静默修复 |
| 长句截断 | 当前数据 0；人工 300 字符句真实 Tokenizer 下丢 46 字符、影响 1 实体 | 当前固定数据无截断事故；一般输入会静默丢失 | 需要覆盖/超长检查；当前无须立即建设滑窗 | A 协调通用策略；本人模型入口加断言 |
| CRF Mask | 是，CLS 首位 False、attention 含特殊项；实际内部空洞为 0 | 直接传普通 CRF 不正确 | 否，不要求公共 DataLoader 输出 CRF 专用 Mask | 本人局部压紧并保留逆映射 |

**A 类：**边界投影、静默修正、一般输入的完整性保障；需要公共维护协调，不能只在 CRF 侧悄悄改出不同的数据政策。
**B 类：**CRF 有效位置、-100 替换、连续 Mask、约束与解码；本人模块负责。
**C 类：**固定划分、标签 ID、原始 ID/原文保留、当前真实数据 offsets 覆盖、CLS/SEP/PAD 的忽略、无标签加载都已通过；不重复建设。

方案比较：

| 方案 | 兼容性与代价 | 标签无损性及风险 |
|---|---|---|
| Token CRF→字符展开 | 可直接消费现有 token labels，加压紧层 | 继承 8 条不可逆边界损失及 2 条非法 Token gold；现有还原含自动修复 |
| 字符 CRF←BERT offsets 表示 | 保留共享 BERT 输入；本人局部恢复字符表示，并另接原字符 gold | 每字符有标签变量，可避免 gold 投影丢失；缺覆盖/重叠须拒绝或明确处理，不能补 O；同词片表示的区分能力待模型验证 |

建议下一阶段优先字符级 CRF。可以先独立做模型结构与人工序列验收，不必等成员 A 修改代码；**正式训练及公平比较前，应先与 A/B 确认共享对齐政策**。若统一采用按原字符组织 Tokenizer 的方案，也必须对训练/推理一致实施，不能用人工实体边界决定分词。

## 6. 命令、证据及复现

证据目录：`docs/bert_crf/evidence/20261019T162653+0800_stage1a/`，下文 E；原始日志位于本轮 `.local/.../logs/`。`commands.jsonl` 记录每个子命令的实际 argv、cwd、起止时间、耗时、退出码、环境变量和日志哈希。

| 实际操作 | 退出码 | 实测耗时秒 |
|---|---:|---:|
| 创建 venv | 0 | 2.742 |
| 官方 CUDA torch 安装 | 0 | 222.269 |
| 其余固定依赖及 pytest 安装 | 0 | 75.748 |
| 首次自检，缺 tzdata | 1 | 8.516 |
| 补装 tzdata | 0 | 4.796 |
| CUDA 自检重试 | 0 | 4.501 |
| 公共 demo | 0 | 6.673 |
| 公共 unittest，6/6 | 0 | 4.561 |
| 全量对齐审计 | 0 | 15.629 |
| pip check | 0 | 0.411 |
| git init -b main | 0 | 0.054 |
| pip freeze --all | 0 | 0.332 |
| 投影后 BIO／无标签输入专项核验 | 0 | 11.779 |
| 报告中的独立 CUDA 复测命令 | 0 | 4.395 |

这些耗时包括相应 Python 子进程启动，不是训练时长。包安装日志、pip --report 原件保留本地，精简下载来源及哈希汇总为 E/installation_sources.json。安装失败为 0；自检启动失败为 1，不能混为 CUDA 算子失败。

核心实际安装命令（解释器均指向 `.venv-bert-crf/Scripts/python.exe`）：

```text
python -m venv .venv-bert-crf
<venv-python> -m pip install torch==2.14.1+cu132 --index-url https://download.pytorch.org/whl/cu132 --report <local>/install_torch_report.json
<venv-python> -m pip install -r <E>/requirements-stage1a.in --index-url https://pypi.org/simple --report <local>/install_dependencies_report.json
<venv-python> -m pip install tzdata --index-url https://pypi.org/simple --report <local>/install_tzdata_report.json
```

上面的 `<...>` 是对已记录绝对路径的缩写，精确 argv 以 E/commands.jsonl 为准。重建最终环境可先安装同一 CUDA torch，再从默认 PyPI 安装 `docs/bert_crf/requirements-stage1a.lock.txt`。该 lock 含最终 tzdata，不能将首次漏装的中间清单称为最终环境。

当前环境轻量复测（repo 根目录，原始输出由 runner 自动保存，不覆盖 GPU 报告）：

```text
python -X utf8 docs/bert_crf/stage1a_runner.py run cuda_recheck -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -c "import torch; torch.manual_seed(1729); x=torch.randn(32,32,device='cuda',requires_grad=True); (x@x).square().mean().backward(); torch.cuda.synchronize(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), bool(torch.isfinite(x.grad).all()))"
```

预期退出 0，输出版本、GPU 名称与 True。GPU/审计采集脚本对已有同名结果拒绝覆盖，重新全量采集应建立新运行目录；`stage1a_runner.py prepare` 也不会重建已有环境状态。上面的直接算子复测不调用固定报告写入。

无需写新结果即可重复公共演示：

```text
python -X utf8 docs/bert_crf/stage1a_runner.py run --cwd C:/userProgram/NLPexp/.local/bert_crf/20261019T162653+0800_stage1a/interface/code demo_recheck -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 -m scripts.demo_dataloader
```

预期退出 0，原始本地日志含“一致: True 未覆盖字符: []”；batch 形状随所选 batch 而定。runner 用过的命令名称不能重复，以免覆盖原始日志。

主要证据：

- E/gpu_smoke.json：真实设备、架构、版本、算子和梯度比较。
- E/batch_checks.json：两份真实 batch、Mask 和压紧探针。
- E/alignment_audit.json、alignment_issues.jsonl：全量统计与无原文定位。
- E/synthetic_tokenizer_probes.json：人工 Unicode/空白/超长边界检查，明确与真实数据分开。
- E/contract_checks.json：Token gold BIO、无标签输入、接口完整性。
- E/tokenizer_manifest.json、input_identity.json、interface_manifest.json：来源与哈希。
- E/execution_summary.json：从原始输出抽取的演示/测试/环境验收结论及收尾完整性。

## 7. 变更、Git 与下一步

先创建 `.gitignore`，排除 `.local/`、`.venv*/`、原始 ZIP、数据/模型/日志/权重/密钥及全部本地证据目录，再执行 git init。当前首次真实源码快照使用实际工作时间，不回填阶段 0 提交、不设远程。具体提交 ID 和状态见实际 `git log -1` 与收尾证据。

本轮新增：`.gitignore`、独立环境和被忽略的本地运行目录、`stage1a_runner.py`、`gpu_smoke.py`、`stage1a_alignment_audit.py`、`stage1a_contract_checks.py`、最终 lock、两份本轮 Markdown 报告及本轮证据。未修改公共接口、名单、原数据 ZIP、课程材料或阶段 0 文件。保留问题定位给维护成员转交，没有代其改代码，也没有向其他人发送消息。

尚未验证：BERT 权重加载和模型前反向、CRF 实现/约束数值验算、完整训练显存和速度、统一预测校验器（尚未交付）、Windows num_workers>0 的 lambda pickle 风险；默认 num_workers=0 已实际可用。

下一轮建议只做最小模型：确认字符级 CRF 路线、共享对齐契约、原字符 gold 的局部接入方式、CRF 实现及约束范围、允许获取的 BERT 权重版本。正式实验配置和分析仍由成员独立确定。当前阶段到此结束，不进入训练。
