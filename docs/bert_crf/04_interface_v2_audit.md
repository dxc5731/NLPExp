# 阶段 1D-0：BERT 数据接口 v2 验收与字符级适配评估

日期：2026-10-20，Asia/Shanghai。运行 ID：`20261020T161159+0800_interface_v2_audit`。

**结论：v2 可用于下一轮受控的真实数据集成；它已消除受审路径中的静默改 BIO、补 O 和截断，但没有解决单个词片表达内部实体边界的限制。必须另建数据适配器和可微字符表示适配层，目前不能直接正式训练。**

本文区分：**实测**（本轮程序执行）、**静态**（源码判断）、**未验证**（未运行）、**待确认**（技术选择）。不计算模型效果，不读取官方 dev/test 金标；没有加载 BERT 权重、执行前后向/优化器、实现 Adapter、修改公共源码或既有模型/训练器。没有 commit/push。

## 1. 版本、材料和运行环境

起始 Git：分支 `dxc`、提交 `663f727949a4b923fa84c3b76ec64efad7bd72a1`、工作区干净。已阅读前五份阶段报告、预测交接规范、现有模型、训练器、验证/恢复契约、测试入口及 v2 文档/源码/测试；课程材料要求的真实证据、严格字符/实体口径、保留集隔离和 AI 使用边界继续遵守。

找到根目录 `BERT数据接口_v2.zip`，实测 SHA-256：

```text
90343bad44ed71bee007a26db7e9f047259c3ad8ab540d659255a731bbcd9c56
```

独立原样解压到 `.local/bert_crf/20261020T161159+0800_interface_v2_audit/interface/`，19 个文件（无目录条目计数）。文件清单及逐文件哈希在 E/`interface_manifest.json`。下文 **E** 为 `docs/bert_crf/evidence/20261020T161159+0800_interface_v2_audit/`，**P** 为上述解压根，**L** 为同次 `.local/bert_crf/<ID>/`。

```text
P/
  README_先看我.md
  doc/{数据接口说明.md, 数据接口说明_v2.md}
  code/
    requirements.txt
    artifacts/{split_v1.json, alignment_report_train.json,
               alignment_report_dev.json, alignment_report_test.json}
    scripts/demo_dataloader.py
    src/{__init__,config,data_io,labels,split,align,dataset,model,metrics}.py
    tests/test_align.py
```

相对 v1：新增 3、修改 9、未变 7、删除 0（E/`v1_v2_diff.json`）。新增 Softmax `model.py`、`metrics.py` 和版本说明；修改 align、dataset、demo、测试、文档和三份对齐报告。**固定 split、标签表、依赖清单、路径配置、原始读取及划分脚本未变。** 未运行划分脚本，未导入/运行 Softmax 模型或计算其指标。dev/test 附带报告只计算文件字节哈希，不解析其中标签明细，不将其成绩用于决策。

本轮复用 `.venv-bert-crf`，未安装、升级或卸载依赖：Python 3.13.5、torch 2.14.1+cu132、Transformers 5.20.1、tokenizers 0.23.3、numpy 2.4.6、huggingface-hub 2.2.0、seqeval 1.2.2、scikit-learn 1.9.1；Windows 11，RTX 5060 Laptop、CC 12.0、CUDA runtime 13.2、CUDA available=True。`pip check` 实测通过。本轮没有重新跑 CUDA 核函数，不把设备查询当作新的模型 GPU 验收。

v2 仍 pin 旧 torch 2.6.0+cu124；没有为照搬旧清单而降级当前已验收环境。数据路径复用阶段 1A 的 `data/`（仅官方 train），Tokenizer 复用同阶段 `tokenizer/`。Tokenizer 为真实 Fast `google-bert/bert-base-chinese`，revision `8f23c25b06e129b6c986331a13d8d025a92cf0ea`；四个本地文件哈希与固定 manifest 相同。未下载新文件。

数据原 ZIP 哈希仍为 `84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67`；实际 train JSON 为 `07cdd3799087283a5a17de8bb43e71cb159a9a2c99c1cfbe8081847fac358f24`；split 字节哈希为 `497f21cf13d8f5accb83f86ffda16a4a59ba716c064ec347e990a219e8957072`，与 v1 **逐字节一致**。

## 2. 公共演示、测试与真实 Batch：实测

在 P/`code/` 执行文档指定入口；通过子进程环境的 `IMCS_DATA_DIR/BERT_MODEL_DIR` 指向已有资源，UTF-8、Fast Tokenizer、离线模式，`num_workers=0`。没有覆盖默认配置文件或在根目录铺设公共工程。

| 操作 | 结果 | 退出码 | 子进程耗时秒 |
|---|---|---:|---:|
| `python -X utf8 -m scripts.demo_dataloader` | 出现 `一致: True`；仅演示第一句 | 0 | 7.935 |
| `python -X utf8 -m unittest discover -s tests -v` | **13 passed，0 failed**；测试本身 2.934 秒 | 0 | 4.810 |
| 全量内部对齐审计 | 98529 句全部检查完成 | 0 | 23.257 |
| `python -X utf8 -m pip check` | No broken requirements found | 0 | 0.697 |
| 定向复核与完整性核对 | 8 条 ID/缺失位置复核，源码/数据哈希一致 | 0 | 7.856 |

全量脚本内部计时 19.641 秒（不含进程启动和部分导入）。本轮没有未预期命令失败；人工边界检查中的异常是被捕获的明确拒绝，不描述为正式运行故障。未重跑旧 130 项模型测试，避免本轮执行模型或优化器；历史通过数不能充当本轮结果。

| 内部集合 | 对话 | 句子 | 字符 | 首 Batch |
|---|---:|---:|---:|---|
| train | 2225 | 88842 | 1153312 | input_ids/attention_mask/labels `[4,11]`；offsets `[4,11,2]` |
| val | 247 | 9687 | 126049 | input_ids/attention_mask/labels `[4,40]`；offsets `[4,40,2]` |
| 合计 | 2472 | 98529 | 1279361 | 全部来自官方 train 的固定对话名单 |

上述张量均 int64；元信息为 `dialogue_id/sentence_id/text` 列表。所有样本 ID、原文、金标及收集顺序逐条与原始 JSON 比对，名单无重复、无交集，并集等于官方 train 对话集合。没有重新划分。

首 train Batch 有 23 个 -100（15 PAD + 8 CLS/SEP），val 为 81（73 PAD + 8 CLS/SEP）；二者 attention 不等于标签有效 Mask，标签 Mask 首列均 False。另取实际歧义 train Batch `[4,24]`：41 个 -100 中还有 4 个位于真实词片内部；val 歧义 Batch `[1,24]` 含 1 个内部忽略位置。这些真实词片**仍有 BERT 输入与 offsets**，只是 token gold 被忽略。

无标签模式也实测通过：从允许的内部数据构造**完全不含 BIO 字段**的记录，以 `collect_examples(..., require_labels=False)` 和 `build_loader(..., with_labels=False)` 读取。train 4 条、val 3 条，包含歧义样本；无 labels、Dataset.char_labels=None，input_ids/offsets 与有标签读取一致。未运行推理或生成预测。

## 3. 全量对齐结果：实测

审计独立调用真实 Tokenizer（不截断、返回 offsets/特殊项 Mask），与 Dataset 编码逐项比对，再核验原始字符金标、每字符覆盖、词片投影及原始还原。未调用 `to_legal_bio` 处理真实数据，未改金标。

| 检查 | train | val | 合计 |
|---|---:|---:|---:|
| 字符原始 BIO 非法句 | 0 | 0 | 0 |
| Dataset 字符金标原样一致句 | 88842 | 9687 | **98529** |
| 每字符恰好覆盖一次的句子 | 88842 | 9687 | **98529** |
| offset 缺覆盖 / 重复 / 越界字符 | 0 | 0 | 0 |
| Token→字符金标完整往返一致句 | 88835 | 9686 | **98521** |
| 往返不完整句（含 None） | 7 | 1 | **8** |
| 歧义词片 / 对应字符 | 7 / 19 | 1 / 2 | **8 / 21** |
| 有具体错误标签的字符（不计 None） | 0 | 0 | 0 |
| 自动 I→B / 自动补 O | 0 / 0 | 0 / 0 | **0 / 0** |
| 实际截断句 / 字符 / 受影响实体 | 0 / 0 / 0 | 0 / 0 / 0 | **0 / 0 / 0** |
| 内部 -100 空洞句 | 7 | 1 | 8 |

8/98529 句即 **0.008119%** 无法完整往返；21/1279361 字符即 **0.001641%** 的还原值为 None。**None 是标签缺失，不是 Token 覆盖缺失**：这些位置有隐藏表示来源，但单个 token gold 无法表达内部边界。不能把“具体标签错误 0”写成“全部标签往返无损”。

旧版的 11 个错误字符现在变成歧义词片范围内的 21 个 None，计数口径不同；不代表原始金标被更改或异常实体变多。新旧八条 ID 已逐条一致核对；包内 train 报告的 8/21/98521 也被独立复现。

| 内部集合 | dialogue_id / sentence_id | 歧义半开字符区间 |
|---|---|---|
| train | 10034734 / 2 | [5,8) |
| train | 10098865 / 21 | [0,3) |
| train | 10193361 / 25 | [3,6) |
| train | 10206772 / 25 | [3,6) |
| train | 10506585 / 10 | [20,22) |
| train | 10515756 / 10 | [5,7) |
| train | 10731191 / 31 | [3,6) |
| val | 10491432 / 7 | [19,21) |

共 3825 个多字符 Token，1020 个覆盖不完全相同的 BIO 标签；其中 1012 个仍可按 B→I 展开规则无损表示，不能将 B/I 不同一概当异常。真正歧义 8 个：7 个内部有实体起点，1 个跨实体终点，所覆盖区间与 13 个原始实体相交。这个 13 是歧义范围交叉计数，**不是截断影响数或模型失败指标**。

将 token labels 的 -100 简单过滤后仍有 2 句违反 BIO：`10098865/21` 剩余序列句首为 I；`10506585/10` 出现非法 I 前驱。E/`targeted_rechecks.json` 记录定位。这说明“压紧 token labels”也不能替代原始字符金标；字符 CRF 不应走此路径。

最长原始字符 train=199、val=159；最长含特殊项 Token train=201、val=155，均小于 MAX_LEN=256。完整长度直方图和分位数在 E/`alignment_audit.json`。当前 train/val 没有空句、空白字符或非 BMP 字符，所以一般 Unicode 必须另用人工探针验收，不能从当前数据外推。

## 4. 长句、Unicode 与剩余公共风险

以下是**人工文本 + 真实 Tokenizer**，不计入真实数据统计；14 项 Tokenizer 探针及 5 项不合法 helper 输入在 E/`synthetic_probes.json`。它们是独立审计探针，不追加到“13 个公共单测”数量中。

- 254 个汉字得到 256 Token，完整接受；300 个汉字得到 302 Token，`AlignmentError` 明确报错并含定位信息。人工实体跨旧截断边界，但新版整句拒绝，无丢尾输出。包内另有 400 字符长句测试通过。**解决静默截断，不等于支持长句推理/分段。**
- 标点、英文数字、emoji、非 BMP、组合字符的本轮样例完整覆盖、O 金标往返一致。原文 `[CLS]中[SEP]` 中的字面特殊串具有非零 offset，应保留其原字符；不能按 `all_special_ids` 无差别删除。插入的 CLS/SEP 是零宽 offset。
- `A B\t中` 缺覆盖 2 字符；交接协议人工串 `A 😀，𠀀é` 缺覆盖 1 空格；控制字符/NBSP 样例缺覆盖 2 字符；纯空格缺 1。新版 encode 均明确拒绝，不补 O。因此现在安全失败，但仍不能为这些输入输出完整预测。
- 低层 restore 对缺覆盖返回 None，`to_legal_bio` 拒绝 None；原始非法 I 在 restore 中保持原样。显式 `to_legal_bio` 人工例修 2 处，这不是本轮真实金标修正数。字符 CRF 应使用自身硬约束并验证原始解码，**不调用此后处理掩盖非法路径**；文档“导出必须调修复”需明确为其 Softmax 流程建议，不能套用到字符 CRF。
- 空串被 encode 接受为仅 CLS/SEP，原字符长度 0；现有 CRF 明确拒绝空序列。当前数据无空句，未来数据适配器应显式拒绝/记录，不能虚构一个字符。

**剩余 helper 校验缺口（实测人工畸形输入，不是正常全量数据事故）：** P/`code/src/align.py:121` 的 restore 用 zip，不检查标签/offset 等长，短标签列表得到缺失 None；label_id=-1 被 Python 负索引解释为 I-Operation；重叠 offsets 后写覆盖先写。`to_legal_bio:138` 不检查标签白名单，能放行 B-Unknown；`encode_sentence:60` 不检查全局 BIO，句首 I 的人工金标被接受。这些不能支持“所有非法输入都报错”的宽泛承诺。

建议 A 负责公共 helper 的范围/长度/白名单/BIO 校验及文档边界；调用者也应有入口校验。**不阻塞当前合法数据走字符金标 + offsets 的局部集成**，不要求本人修改 A 的源码。当前真实覆盖和金标合法已全量核实，不因畸形探针失败而重构公共模块。

**静态风险：** DataLoader 仍用 lambda collate（dataset.py:144），Windows `num_workers>0` 的 pickle 风险本轮未测试；继续用 0。新增公共 metrics 与本人指标不同：字符 macro 是包含 O 的固定 11 BIO 类，而本人是排除 O 的五实体类型聚合；公共 seqeval 调用未显式指定 strict/IOB2，不能仅凭注释认定非法 BIO 严格拒绝。本轮未执行它、不修改它，后续与 D 对齐口径。

## 5. 接口字段与两类适配

| 内容 | v2 实际提供 | 后续需要 |
|---|---|---|
| 原文、原 ID | Batch 列表，实测原样 | 组合 Trainer 的 `(dialogue_id,sentence_id)` 字符串 tuple |
| 字符金标 | `Dataset.char_labels[i]` 原样；**getitem/Batch 没有** | 依据原 ID 可靠取值，映射 0..10 并右 padding；不能用打乱 Batch 下标直接索引 Dataset |
| 字符长度 | Encoded.n_char，但 Dataset.encoded 没保存；Batch 无长度字段 | 从保留的 `len(text)` 推导，构造连续前缀 bool character_mask |
| input_ids / attention_mask | Batch int64 `[B,L]` | 直接给 BERT；attention_mask 包含 CLS/SEP，不能作为字符 Mask |
| offsets | Batch int64 `[B,L,2]` | 在原 Unicode 码点坐标中校验覆盖并生成未来 mapping 元信息 |
| 特殊项 | 零宽 offsets 与 attention 可识别；Batch 无 special_tokens_mask | 当前全量与 Tokenizer 显式 Mask 已比对一致；排除插入项/PAD，不按 gold -100 排除真实词片 |
| char→token 表 / 覆盖标记 | **无显式字段**；encode 已强制完整覆盖、无重叠 | 可由 offsets 构造；策略版本写入 metadata，不必让 A 再做另一套预处理 |
| 截断检测 | 关闭截断并按 max_len 报错 | 保持 fail-fast；未来需要超长预测时再确认完整分段/还原方案 |
| split / 数据版本 | 固定 split_v1.json；不在 Batch 内 | Provider metadata 记录原包/原 JSON/split/tokenizer/标签/映射哈希 |
| Token/字符隐藏表示 | **均不是数据接口输出** | BERT 运行才产生 `[B,L,H]`；真实 Adapter 未来产生 `[B,C,H]` |

完整机器可读映射表见 E/`field_mapping.json`。

**静态核实：现有模型没有真实字符表示转换。** `src/bert_crf/model.py` 的 `CharacterRepresentationAdapter` 是抽象类；组合模型 forward 明确调用 `self.adapter(tokens, character_mask=..., **mapping)`。已有一对一 gather 位于人工测试路径，不能套用真实的多字符 Token。模块注释仍提到“等待 1C”是历史阶段措辞，实际进度仍是未实现；本轮没有改它。

因此同时需要：

1. **数据适配器 / BatchProvider**：封装现有 DataLoader 的 ID、BERT 输入、原始字符标签、字符 Mask 和映射元信息；维护打乱后 ID→gold 对应与 Epoch 专用种子，不重写公共加载/划分。
2. **字符表示适配层**：接收 BERT token hidden states，按同一 mapping 输出完整 `[B,C,H]`。它是模型中的可微操作，和标签转换是两件事。

## 6. 推荐的最小表示方案：静态分析，待确认

| 方案 | 兼容性及代价 | 风险 / 处理 |
|---|---|---|
| offsets 展开，一个 Token 向覆盖的每个原字符共享 hidden | 当前全量每字恰好一个来源，可用无参数 gather；不改 Tokenizer，不依赖金标 | 同词片字符共享内容表示，eval 时分类发射可能相同，内部边界辨别能力有限；CRF 不保证恢复正确边界，效果待未来真实实验 |
| 一个字符多个 Token 时按均值聚合 | 需显式版本化；理论可微，避免随意取首个 | 当前 v2 encode 拒绝重叠，全量未出现这种情况；仅未来策略候选，不默认放行重叠 |
| 为缺覆盖字符设计独立可训练 fallback/字符特征 | 可保留所有原字符变量，需额外特征/位置及 checkpoint/schema 约定 | 会改变结构、公平比较和可辨性，不能当成已有 BERT 表示；须单独设计与验收 |
| 更改 Tokenizer 为可完整表示的字符/片段策略 | 可能解决空白与粒度问题，但要保持原文坐标与训练/推理一致 | 需要 A/B 共同确认；不能按 gold 实体边界分词，不能悄悄改变公共方案 |

推荐下一轮最小基线：**对完整覆盖且不重叠输入，按 offsets 共享 Token 表示；全部原字符用原金标参与 CRF；未覆盖/重叠/越界/超长/空串先明确拒绝。** 对这次全量内部数据，此拒绝策略不会删掉任何句子；8 个歧义词片仍参与表示映射，其 21 个字符也仍保留金标监督。

gather、共享与均值在数学上可微，共享位置的梯度会累加到源 Token；这只是机制分析。**本轮没有实现或测试 v2 梯度链路**，不能用阶段 1B 的一对一通过替代未来多字符映射验收。baseline 非参数化可兼容现有独立模型保存限制；若改用可训练 fallback，需另审其 `save_pretrained` 对 Adapter 参数的拒绝及训练 Checkpoint 身份策略。

## 7. 与阶段 1C 框架的衔接：仅静态

现有训练循环、AdamW、验证指标和 Epoch 恢复机制可复用，暂不需要改 CRF 规则、标签范围或字符 Mask 校验。BatchProvider 需要 `size/metadata/batches(epoch,batch_size,shuffle,seed)`，每批 `ids` 和 `inputs`；inputs 对应模型 `input_ids, attention_mask, character_labels, character_mask, mapping`。单句 BERT 可省略 token_type_ids。

适配器应保证 Padding 的标签占位是合法 0（O），只对 Padding Mask=False；这是张量占位，不是给缺覆盖原字符补 O。不能将 token labels=-100 传入 CRF，也不能由 `labels != -100` 生成字符 Mask。批次构建、mapping 和推理必须使用同一纯文本/offset 规则与版本，推理不得读取 gold。

Trainer 当前 `require_synthetic`、Provider 类型检查及正式 `train.py` **仍然锁定真实执行**；未来集成验收后需明确授权开启真实模式及建立正式配置/元数据校验。这是执行闸门工作，不是 CRF 算法修复。本轮未解除闸门、未把真实 Provider 冒充人工。

Checkpoint 身份包含模型/Adapter 类、Provider metadata 和配置；将真实映射策略/版本及源哈希写进 metadata，才能拒绝不同规则间恢复。无参数共享方案可复用现有状态格式；具体真实 Provider 随机顺序和恢复一致性仍需集成后实测。

将来可从字符 CRF 的严格解码得到每原字符一标签，按保留的对话 ID/sentence_id 写 UTF-8 JSON；仍需独立执行完整 ID 集合、长度、标签白名单、BIO、特殊项和长句还原校验。**当前没有保证最终交接已通过**：真实 Adapter、正式推理/导出均未实现，规范所述 `src/validate_predictions.py` 本工作区仍不存在。不得采用 v2 的 `to_legal_bio` 修正后宣称 CRF 解码原本合法。

## 8. 证据、复现与修改范围

E/ 主要文件：

- `run_context.json, input_identity.json, interface_manifest.json, v1_v2_diff.json`：路径、版本和逐文件哈希。
- `environment.json, initial_git.json, source_manifest.json, execution_summary.json`：环境与源码完整性；140 个既有受跟踪文件、v1 16 文件和 v2 19 文件本轮均逐字节未改。
- `commands.jsonl`：实际 argv/cwd/环境变量、时间、退出码和原始日志 SHA-256。
- `alignment_audit.json, alignment_issues.jsonl`：全量统计及八条无原文定位。
- `batch_checks.json, synthetic_probes.json, targeted_rechecks.json, field_mapping.json`：真实 Batch、人工边界、定向复核及静态衔接表。

原始日志在 L/`logs/{demo,supplied_tests,full_alignment,dependency_check,integrity_and_targeted_checks}.log`。demo 含真实医疗原文，只在受 Git 忽略的本地目录保存，不复制到报告或公开证据。其他结构化诊断只保存必要 ID、位置、标签和原文哈希，不保存医疗原文/可还原的 token 序列；未调用 Dataset.report 的原文样例输出。源包自带的原文细节也仅位于忽略的原样解压目录。

根目录已实际执行的命令（venv Python 路径与 runner 的完整参数见 commands）：

```text
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/interface_v2_runner.py prepare
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/interface_v2_runner.py run demo --interface -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 -m scripts.demo_dataloader
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/interface_v2_runner.py run supplied_tests --interface -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 -m unittest discover -s tests -v
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/interface_v2_runner.py run full_alignment -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/interface_v2_alignment_audit.py
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/interface_v2_runner.py run dependency_check -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 -m pip check
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/interface_v2_runner.py run integrity_and_targeted_checks -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/interface_v2_finalize.py
```

prepare 只运行一次，现有 state、日志及 JSON 均拒绝覆盖；其创建结果保留在 run_context 和本会话实际输出。重跑演示/单测可使用新命令名；重跑完整审计应先另建全新运行状态/目录，不能覆盖此次结果。预读/Git 查询的原始输出在本会话，核心验收子命令的文件日志齐全。

本轮新增（未暂存、未提交）：`docs/bert_crf/04_interface_v2_audit.md`、`interface_v2_runner.py`、`interface_v2_alignment_audit.py`、`interface_v2_finalize.py`。新增的独立审计解压目录、状态和 E/ 受现有 `.gitignore` 排除，未修改 `.gitignore`，未生成公开副本；以后如批准提交，应另行选取无原文轻量证据，不把原始 demo 日志一起加入。最终收尾清单见 E/`final_status.json`。

Codex 辅助阅读、审计脚本、运行、版本核对和工程说明；统计来自实际程序。没有代写正式实验设计/结论或贡献声明，没有向 A 或其他人发送消息。

## 9. 阶段判断与下一步（不执行）

| 条件 | 判断及责任 |
|---|---|
| 公共加载、固定划分、原字符金标 | 实测通过，可复用 |
| 静默 BIO/补 O/截断 | 本轮受审路径已修复，变为原样返回/显式缺失/拒绝；不是通用长句或 Unicode 输出方案 |
| 词片内部边界 | 8 句仍不可逆，A 已显式标记；本人字符方案使用原 gold 和所有 offsets，不必要求 A 把歧义 token gold 伪造成可逆 |
| 公共辅助校验、说明与指标口径 | A/B/D 协调明确范围、strict 验证和公平比较；不阻塞受控内部字符集成 |
| 数据 Provider 与真实字符表示 | 均未实现，由本人下一轮在独立模块完成 |
| 正式执行与最终交接 | 未具备；需映射测试、真实小 Batch 链路、恢复/推理一致性、字符输出校验及本人批准 |

下一轮最小验收建议：确认 offsets 共享 + 缺覆盖拒绝基线；实现 ID 安全的 BatchProvider 和无参数字符表示层；用八条歧义样例及人工 Unicode/长度异常验证原字符 gold 不变、完整解码长度、特殊项排除、梯度与保存重载，再进行极小真实 Batch 的集成验证。应将“有数据可集成”与“允许正式训练”分别验收。本轮在审计结束后停止，等待本人确认。
