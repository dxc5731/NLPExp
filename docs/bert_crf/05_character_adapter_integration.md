# BERT-CRF 阶段 1D-1：字符级适配与真实数据集成验收

日期：2026-10-20，Asia/Shanghai。运行 ID：`20261020T164334+0800_stage1d1`。
工作区 `C:/userProgram/NLPexp`，分支 `dxc`，HEAD `663f727949a4b923fa84c3b76ec64efad7bd72a1`。

**结论：本轮字符级适配验收通过。** 新适配器重新扫描全部内部 train/val，98529 句、1279361 字符完整且唯一覆盖，原金标全部保持一致；8 条歧义样本均能保留各字符标签并通过真实预训练 BERT→字符 CRF 的有限 Loss 与合法解码检查。真实 train 小 Batch 完成 GPU backward，没有优化器或参数更新。最终 **188 passed，0 failed，0 skipped＝原有 130 + 新增 58**。

这是 `real_data_integration_test`，不是正式微调、选模或效果实验。真实数据没有计算 F1、导出预测或产生训练成绩。回归测试中的参数更新及恢复仍仅使用人工样例。以下“实测”仅指本轮实际程序验证；后续效果和真实训练恢复能力另列为未验证。

证据目录 E：`docs/bert_crf/evidence/20261020T164334+0800_stage1d1/`。
本地目录 L：`.local/bert_crf/20261020T164334+0800_stage1d1/`。

## 1. 实际代码成果与接口

本轮只新增九个文件，没有修改既有模型、CRF、训练器或公共源码：

| 文件 | 职责 |
|---|---|
| `src/bert_crf/data_adapter.py` | `V2DataAdapter`、`CharacterBatch`、`V2BatchProvider`；组合 ID 查原 gold，核对编码并输出模型/Trainer 所需字段 |
| `src/bert_crf/character_mapping.py` | `OffsetCharacterAdapter`、覆盖检查、结构化 `MappingError`；无参数可微 gather |
| `tests/bert_crf/test_character_adapter.py` | 49 项人工输入测试，使用真实 Fast Tokenizer、小随机 BERT，验证精确表示/梯度、拒绝行为和保存重载 |
| `tests/bert_crf/test_character_real_integration.py` | 9 项显式授权的真实内部数据验收，8 条逐样本检查和覆盖/GPU证据核对 |
| `configs/bert_crf/stage1d1_integration.json` | 本次 CUDA/Batch/映射/拒绝/零更新约束，明确标记运行性质 |
| `docs/bert_crf/stage1d1_runner.py` | 独立目录、环境隔离、命令/退出码/原始日志记录 |
| `docs/bert_crf/stage1d1_integration_check.py` | 全量轻量扫描、8 样本复核及真实小 Batch 前后向/重载验收；没有优化器和指标调用 |
| `docs/bert_crf/stage1d1_finalize.py` | JUnit 计数、旧文件和输入哈希、源码快照与最终 Git 状态 |
| 本报告 | 技术过程与验收边界 |

复用现有 `CharacterBertCRF(config, BertEncoder, OffsetCharacterAdapter())`，无需再创建重复的集成模型。数据流为公共 Dataset/Collate → 新数据适配器 → 现有 BERT → 新字符映射 → 现有 Dropout/Linear/CRF。总参数仍为 **101685658**，新映射没有新增可训练参数。

`CharacterBatch` 保留原 IDs、原文和下列张量：`input_ids/attention_mask [B,L]`、`offsets [B,L,2]`、`character_labels [B,C]`、连续前缀 bool `character_mask [B,C]`、long `character_lengths [B]`。另提供 bool `special_tokens_mask`。`model_inputs()` 返回原模型签名，`trainer_batch()` 返回 `ids + inputs`，没有向旧模型传入额外元信息。

映射策略固定为 **`offset_share_unique_v1`**、坐标单位 **`python_unicode_codepoint`**：Token `[s,e)` 的表示由所有 `s..e-1` 原字符共享，每字符必须只有一个来源。无均值聚合、邻近复制、标签修复或缺字符回退。特殊项由真实 Fast Tokenizer 的 `special_tokens_mask` 和零宽 offsets 判断；PAD 必须无区间。原文字面 `[CLS]`/`[SEP]` 有非零 span 时保留，不能按 Token ID 删除。

Adapter 先用非 BMP 探针确认 Python 字符坐标，再对 Batch 原文重新调用同一 Tokenizer（不截断），逐项比对 Dataset 快照和 Batch 的 IDs/offsets/attention；没有仅依赖调用方写一个 offset 单位字符串。映射层再次检查形状、dtype、区间、覆盖和字符 Mask，用 gather 后清零字符 Padding。人工精确测试验证共享位置的梯度累加到源 Token，CLS/SEP/PAD 不获得映射损失梯度。

缺覆盖、重复覆盖、逆序、非法区间、空串、Mask/长度/编码不一致均明确拒绝；异常只含结构化原因、位置或 ID，不输出原文。当前完整内部数据没有触发这些拒绝。

## 2. 原始字符金标与八条歧义

**实测：**唯一金标来源是 v2 的 `Dataset.char_labels[i]`。构造适配器时按 `(dialogue_id,sentence_id)` 建立唯一字典，复制原文、字符标签和编码身份；拒绝重复组合 ID、非字符串 ID、缺标签、长度不等、标签不在固定 11 类或非法 BIO。核对 `Dataset.examples[i].tags`，全量审计另外逐条核对原始 JSON 的 `sentence/BIO_label`。打乱 Batch 后仍按组合 ID 查找，不用当前 Batch 下标或单独句子 ID。

公共 Token `labels` 完全不参与字符监督或 Mask 构造；人工测试把全部 Token labels 设为 -100、甚至换成无关对象，字符 gold 不变。字符 Padding 使用合法 O=0，仅作为 Mask=False 的占位；每个有效原字符都有原 gold，没有为缺覆盖字符补 O。固定 11 标签顺序与旧 CRF 配置完全一致。

八条定位直接读取 D0 的真实 `alignment_issues.jsonl`（路径及 SHA-256 收于 E/input_identity.json），并核对每句原文哈希、长度、歧义 span 中的原始标签和唯一 Token 索引。

| 内部集合 | 原 dialogue_id / sentence_id | 字符长度 | 歧义 span | 本轮结果 |
|---|---|---:|---|---|
| train | 10034734 / 2 | 12 | [5,8) | 原 gold 保留；唯一覆盖；有限 NLL；合法等长解码 |
| train | 10098865 / 21 | 9 | [0,3) | 同上 |
| train | 10193361 / 25 | 24 | [3,6) | 同上 |
| train | 10206772 / 25 | 25 | [3,6) | 同上 |
| train | 10506585 / 10 | 28 | [20,22) | 同上 |
| train | 10515756 / 10 | 18 | [5,7) | 同上 |
| train | 10731191 / 31 | 17 | [3,6) | 同上 |
| val | 10491432 / 7 | 24 | [19,21) | 同上 |

E/`eight_cases_data.json` 保存原文/金标哈希与共享区间核对，E/`eight_cases_model.json` 保存八条真实 NLL、解码长度及 BIO 合法性。没有改任何金标、没有跳过样本。**解决的是字符监督完整进入 CRF；不宣称同 Token 内部边界已被模型正确识别。** 原 Token 标签仍无法表达这些内部边界，并未修改公共转换策略。

## 3. 全量真实覆盖审计

仅从官方 **train JSON** 读取固定内部 train/val；未读取官方 dev/test 标签，没有重划分。公共包复用 D0 已解压的独立只读快照。

| 检查 | train | val | 合计 |
|---|---:|---:|---:|
| 对话 | 2225 | 247 | 2472 |
| 句子 / 全部接受句 | 88842 | 9687 | **98529** |
| 原字符 / 恰好覆盖一次字符 | 1153312 | 126049 | **1279361** |
| 原字符金标完全一致句 | 88842 | 9687 | **98529** |
| 缺口字符 / 重叠字符 | 0 / 0 | 0 / 0 | **0 / 0** |
| 非法 offset 句 | 0 | 0 | **0** |
| 标签长度不等 / 非法原 BIO 句 | 0 / 0 | 0 / 0 | **0 / 0** |
| 被拒绝句 | 0 | 0 | **0** |
| 多字符 Token | 3440 | 385 | 3825 |
| 完全不含标签的读取核对样本 | 9 | 3 | 12 |

本轮重新运行新适配器，并逐字符核对 gather 索引所指区间，没有复用历史统计冒充新审计。检查源码不调用标签还原或 `to_legal_bio`，没有 I→B 或补 O 路径。覆盖缺口、重叠和拒绝率均为 0；8 条 Token 歧义未造成字符表示缺失。

全量进程耗时 **127.716 秒**，含导入、真实 Tokenizer、Dataset、数据身份、逐字符索引、无标签读取及 Provider 入口检查；不是模型推理耗时。机器可读结果为 E/`coverage_audit.json`、`batch_contract.json`。没有对全量数据运行 BERT。

## 4. 真实 GPU Batch 集成

本轮实际环境复核：Python **3.13.5**，PyTorch **2.14.1+cu132**，Transformers **5.20.1**，tokenizers **0.23.3**，CUDA runtime **13.2**；RTX 5060 Laptop，Compute Capability **12.0**。独立 `.venv-bert-crf`，没有安装或升级依赖。

模型来源仍为 `google-bert/bert-base-chinese`，revision `8f23c25b06e129b6c986331a13d8d025a92cf0ea`。复用 `.local/bert_crf/20261019T174356+0800_stage1b/pretrained/`；权重 SHA-256：`3404a1ffd8da507042e8161013ba2a4fc49858b4e3f8fbf5ce5724f94883aec3`。加载日志的多余 MLM/NSP/pooler 参数来自去除原预训练头的 encoder 加载，原核心回归另检查无缺失/尺寸错配并逐张量对照源权重。

| 小 Batch 实测 | train | val |
|---|---|---|
| BERT input_ids / attention_mask，int64 | [2,10] | [2,40] |
| offsets，int64 | [2,10,2] | [2,40,2] |
| 字符 labels int64 / Mask bool | [2,8] | [2,39] |
| 原字符长度 / 预测长度 | [2,8] / [2,8] | [39,5] / [39,5] |
| 序列均值 NLL | **11.773172378540039** | **53.44755554199219** |
| 执行 | train mode 前向、解码、backward | eval/no_grad 前向、解码 |
| 同步耗时 | **0.738567 秒**，含前向/解码/backward | **0.033108 秒**，含前向/解码 |

这两个 NLL 仅用于有限性与链路验收，不能当正式训练/验证成绩。val 未计算 F1、未选模。八条歧义额外逐句执行实际预训练模型的 eval Loss/解码，全部有限、长度正确、严格 BIO 合法。去掉 val 的字符标签后解码结果相同，未调用预测修补函数。

梯度实测：BERT **197/197**、Linear **2/2**、CRF **3/3** 参数张量全部获得有限且非零梯度。真实数据路径没有创建优化器；`optimizer_steps=0`。运行前后所有参数字节 SHA-256 相同：`e09c1165d7e6e7291dddc24741b141da9a270f1214dfec0388761e98f2498f71`，所有参数版本计数也相同。

预训练加载并搬到 CUDA **1.508819 秒**；峰值 allocated **997216768 bytes（951.02 MiB）**、reserved **1082130432 bytes（1032 MiB）**，范围为本进程模型加载及小 Batch 验收、重载前，不是整卡 `nvidia-smi` 使用量。脚本主体 **9.157891 秒**，外层进程 **15.885655 秒**，后者含解释器/导入。

使用同一真实 Adapter 保存到 L/`integrated_checkpoint/` 并重载，eval 发射分数逐位相同、路径相同、参数哈希相同。重载及复核 **1.180681 秒**。Checkpoint 只在 Git 忽略目录，策略版本另存 E/JSON/配置；现有 `save_pretrained` 仍需调用方显式提供同一 Adapter，不会自动检测任意不同实现，后续正式推理应校验这些元数据。

本轮未发生实际 OOM 或真实数据集成故障。人工故障注入的拒绝测试不是实际 GPU 故障。

## 5. 自动测试、命令与真实修复

| 文件化命令名 | 结果 | 退出码 |
|---|---|---:|
| `new_tests_01` | 48 passed，1 failed | 1 |
| `new_tests_02` | 49 passed，0 failed | 0 |
| `full_coverage_01` | 全量新适配覆盖和原 gold 核对通过 | 0 |
| `real_cuda_01` | 真实小 Batch、八样本、零更新和重载通过 | 0 |
| `full_regression_01` | **188 passed，0 failed，0 skipped**；pytest 20.75 秒 | 0 |
| `finalize` | 源码/输入完整性与 JUnit/日志溯源，见 final_status | 运行记录见 commands.jsonl |

唯一实际单测失败：`test_batch_rejections[offset]` 假定 `abc` 的原终点不是 3；但真实 Tokenizer 的 offset 正是 `[0,3)`，赋值 3 没有损坏输入，所以没有报错。修复本人测试为“原终点加一”，未修改金标或公共代码；首轮 XML/原始日志与成功重跑均保留。随后静态补充 Dataset `with_labels` 与字符金标可用性一致性检查、映射 offsets 类型检查，最终完整回归验证的是最终代码。

49 项人工测试覆盖精确向量和共享梯度、特殊项/PAD、复合 ID/打乱、标签白名单和 BIO、缺标签、offset/shape/dtype/Mask 异常、真实 Fast Tokenizer 的 Unicode/字面特殊串/空白拒绝、小随机 BERT→CRF 与保存重载。9 项真实验收覆盖八例及全量/GPU证据。旧 130 项未改，CRF 枚举、BIO 训练/解码硬约束、CPU/CUDA、指标、Checkpoint/恢复、异常和训练闸门均通过。

特别说明：`test_trainer_pretrained` 重验保留的阶段 1C 人工训练器执行证据、Checkpoint 哈希及日志，没有重新执行该真实 BERT 人工优化流程；其余轻量人工 Trainer 更新与 CPU/CUDA Epoch 恢复测试本轮实际运行。不能将“旧 130 全通过”解释成已经做过真实数据训练或真实数据续训。

旧 `test_pretrained` 的人工一对一 fixture 仍保留历史字段 `real_character_adapter=NOT_IMPLEMENTED`，其含义仅是该旧测试路径不使用真实适配；不能作为本轮项目状态。当前真实 Adapter 的独立验收证据是 `real_batch_integration.json`，既有测试源文件未因阶段措辞而改写。

核心实际命令在根目录执行，完整 argv、环境变量、cwd、时间、退出码和原始日志哈希见 E/`commands.jsonl`：

```text
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d1_runner.py prepare
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d1_runner.py run new_tests_02 -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 -m pytest tests/bert_crf/test_character_adapter.py -q --basetemp=.local/bert_crf/20261020T164334+0800_stage1d1/pytest_new_02 --junitxml=docs/bert_crf/evidence/20261020T164334+0800_stage1d1/new_tests_02.xml
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d1_runner.py run full_coverage_01 -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d1_integration_check.py scan
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d1_runner.py run real_cuda_01 -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d1_integration_check.py cuda
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d1_runner.py run full_regression_01 -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 -m pytest tests/bert_crf -v -s --basetemp=.local/bert_crf/20261020T164334+0800_stage1d1/pytest_all_01 --pretrained-dir=.local/bert_crf/20261019T174356+0800_stage1b/pretrained --smoke-output=docs/bert_crf/evidence/20261020T164334+0800_stage1d1/core_gpu_regression.json --checkpoint-dir=.local/bert_crf/20261020T164334+0800_stage1d1/core_regression_checkpoint --trainer-smoke-summary=docs/bert_crf/evidence/20261019T192215+0800_stage1c/pretrained_trainer_final.json --junitxml=docs/bert_crf/evidence/20261020T164334+0800_stage1d1/full_regression_01.xml
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d1_runner.py run finalize -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d1_finalize.py
```

不要原样覆盖这些输出。完整复跑先使用 runner 的 `--state .local/bert_crf/<新的状态名>.json prepare` 建立独立状态；之后每次 `--state <同一路径> run ...`。根据新 state 中的 evidence/local 路径指定新的 pytest 输出及临时目录。预备查询的输出保存在会话；核心验收命令都保存文件日志，没有把未文件化查询伪称独立日志。

## 6. 证据、Git 与完整性

v2 原 ZIP SHA-256：`90343bad44ed71bee007a26db7e9f047259c3ad8ab540d659255a731bbcd9c56`。
官方 train JSON SHA-256：`07cdd3799087283a5a17de8bb43e71cb159a9a2c99c1cfbe8081847fac358f24`。
本轮固定 split、Tokenizer、权重、历史八例及公共快照均复用已审计来源，原始字节哈希见 E/`input_identity.json`、`protected_manifest.json`；收尾再次核对。

| 证据 | 内容 |
|---|---|
| `environment.json, integration_config.json` | 实际环境与运行配置 |
| `initial_git.json, protected_manifest.json, input_identity.json` | 基线和原件哈希 |
| `coverage_audit.json, batch_contract.json` | 全量计数、字段/类型及 Provider 元数据 |
| `eight_cases_data.json, eight_cases_model.json` | 八例原 gold、映射与实际模型验收 |
| `real_batch_integration.json` | 小 Batch NLL、梯度、显存、时间、参数未变及重载 |
| `new_tests_01.xml, new_tests_02.xml, full_regression_01.xml` | 首次失败、修复和完整回归 |
| `commands.jsonl, test_summary.json, known_issues.json` | 实际执行、计数和限制 |
| `source_manifest.json, source_snapshot/, final_status.json` | 未提交源码定位、完整性与最终 Git 状态 |

原始命令日志在 L/`logs/`。上述结构化证据只保存 ID、位置、标签/原文哈希和工程诊断，不保存医疗原文、可还原 Token ID 列表或真实预测序列。大 Checkpoint/模型、数据和环境留在忽略目录，没有下载新权重、上传数据或联系队友。

工作区保留原有四个 D0 未提交文件，本轮另新增九文件；没有暂存、commit 或 push。HEAD 没有变化，工作区因这些未跟踪文件为 dirty。收尾逐字节验证全部原受跟踪文件、D0 四文件及公共 v2 快照未改，结果由 `final_status.json` 给出。E/L 受既有 `.gitignore` 排除；本报告保留轻量计数/命令，脚本和源码可供之后人工审核提交，未自动强制加入原始证据。

Codex 辅助了代码、人工测试、运行工具、调试和工程验收说明；数值来自实际程序。没有代写本人正式研究结论、实验动机或贡献声明。正式比较设计与效果分析仍由本人完成、确认。

## 7. 后续判断与未验证边界

**已实测具备：**当前完整内部数据可进入新字符级适配路径；真实预训练 BERT 和旧 CRF 核心兼容；八例监督无损；前后向、等长合法解码和本机保存重载通过。不需要先让 A 修改这些八条原 gold，也没有发现阻塞当前内部数据的真实 offset 问题。

**仍未验证/未开放：**真实数据优化器更新、完整 Epoch、显存预算、真实 Provider 的训练续跑和性能均未执行。`Trainer` 和 `train.py` 继续拒绝真实训练；新 Provider 明确标记 `real_data_integration_test`，没有冒充人工 Provider 绕过闸门。通用空白/控制符缺覆盖、空串和超长输入仍明确拒绝，不具备通用完整平台推理/长句还原保证；最终预测导出及统一交接校验未开发或执行。

**方案局限：**同一个 Token 内字符共享初始特征，eval 的 Linear 发射也可能相同，CRF 转移约束不能保证识别正确内部边界；本轮仅确认原 gold 可以完整监督，不能宣称模型质量改善或这是最优映射。

**建议下一轮最小任务（本轮不执行）：**本人确认首次正式配置、数据/模型/映射元数据和实验编号后，单独建设并测试受控真实训练入口；核验新 Provider 的顺序/恢复身份、最佳模型加载和训练/推理策略检查，再按明确授权开展第一次受控微调。正式启动前与 A/Softmax/D 统一原字符评价口径和比较协议：CRF 保留歧义范围的全部字符监督；若 Softmax 采用公共 Token labels 计算交叉熵，则会忽略这些歧义词片，需如实说明差别。Softmax 的实际训练配置和数据调用尚待成员核实，不能从随包模型构造函数推断其最终实验实现。A 的一般 helper 校验/空白和长句边界仍按 D0 建议协调，不由本人本轮更改。

本阶段验收结束，停止于真实数据集成，不自动开始正式训练或下一阶段。

## 8. 与 BERT-Softmax 的监督粒度差异及最终评估注意事项

本节在阶段 1D 收尾审核时补充（2026-10-20），不改写上面的历史运行结果。依据本人 `src/bert_crf/data_adapter.py`、`character_mapping.py`、`model.py`、v2 原样快照的 `code/src/{model,dataset,align}.py` 及随包说明进行核对。

**已实测的本 BERT-CRF 路径：**监督直接来自原始字符级 BIO 金标 `Dataset.char_labels`。BERT 首先产生 Token 级隐藏表示，`offset_share_unique_v1` 根据原文半开 offsets 将表示共享给覆盖的每个原始字符。现有 CRF 在完整字符序列上计算负对数似然和 Viterbi 解码，输出长度等于原文 Python Unicode 码点数；不使用 Token labels 作为 CRF 金标，不对原始 BIO 做修正。原有 8 句 Token 投影不能无损表示内部边界；本模型保留其全部原始字符标签，避免依赖有损投影。这不等于已经正确预测这些边界，也不说明模型效果优于其他方法。

**Softmax 的证据边界：**v2 包中的 `code/src/model.py:17` 使用 `AutoModelForTokenClassification` 构造 11 类 Token 分类模型；`dataset.py` 返回 Token 级 `labels`，`align.py` 将无法无损表示原字符标签的词片标为 -100，说明文档约定由交叉熵忽略。这些是包内实现及使用建议的**静态事实**。当前未交付可核验的 Softmax 正式训练入口、最终配置、调用日志或 Checkpoint，因此“Softmax 成员的实际实验是否使用这一路径、是否另做字符表示映射或字符监督”仍为**待核实**，不能把包内默认方案当作其最终实验事实。

如果 Softmax 最终采用上述 Token labels 监督，则两种方法同时存在监督粒度和表示处理差异：字符 CRF 监督所有有效原字符，Token Softmax 对歧义词片的 Token loss 不监督，并需另行还原字符预测。即使 BERT、内部划分和评价数据相同，最终差异也不能全部归因于 CRF。请最终评估成员保留下列检查项：

- 各方法保留原始对话 ID、句子 ID 和字符顺序，每个原始字符输出一个 BIO 标签，包含标点和空白；不把特殊 Token 或 Padding 当原字符，不改 ID、不删句、不修金标。
- 使用统一的 11 标签映射、严格 BIO 合法性检查和严格实体边界匹配规则；实体必须同时匹配句子身份、起点、半开终点及类别，统一实体级 Precision/Recall/F1 的 Micro/Macro、零分母和统计范围。字符级指标与实体级指标分别报告。
- 记录每种方法实际的监督粒度、字符映射策略/版本、歧义或未覆盖字符处理、截断/分段规则，以及预测后是否修复 BIO、实际修改数量；不能用修复后的合法序列冒充原始解码已合法。
- 若监督粒度不同，应将分数解释为**完整方法之间的比较**，不能直接声称性能差异完全来自 CRF；在统一评价前先核实 Softmax 的实际实现，不要求其他成员本轮返工。
- 后续若要单独研究 CRF 的贡献，应由本人/团队设计控制实验，尽量固定 BERT、字符映射、训练数据、训练条件和评价规则，仅改变输出层。本轮未设计或执行这类正式实验，也未填写效果结论。

交接仍遵循《预测结果交接规范》：完整 ID/字符级输出校验、独立运行目录和真实模型/代码/数据/配置/耗时元数据；人工测试记录不能当正式 `run_info.json`。本节只是后续评价提醒，不证明最终统一评估和预测交接已经完成。
