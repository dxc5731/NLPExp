# 阶段 1D 收尾审核与正式微调前置条件检查

日期：2026-10-20，Asia/Shanghai。工作区：`C:/userProgram/NLPexp`。
独立复核运行：`20261020T171714+0800_stage1d_closeout_review`。

**最终结论：不通过本轮“可直接开始首次正式微调”的收尾标准；不提交、不推送。** 数据适配和模型集成复测通过，但正式训练入口、真实训练 Provider 模式及独立正式配置尚未接通。下一步仍需要实质性代码集成，不能仅凭本人的训练授权或改一个配置就执行，因此不满足本轮通过标准。这不是 CRF 或原始数据验收失败，也不是只缺准确文档。

本报告不把“正式训练仍需本人授权”视为问题；实际阻塞是：即使未来获得授权，现有代码也无条件拒绝真实训练，并没有可使用的正式启动路径。本轮没有绕过保护、修改训练逻辑或启动真实优化器。

E：`docs/bert_crf/evidence/20261020T171714+0800_stage1d_closeout_review/`。
L：`.local/bert_crf/20261020T171714+0800_stage1d_closeout_review/`。
所有判定标明实测、静态或未验证；结构化结果、完整 argv、环境与原始日志哈希在 E，原始日志仅在 L。

## 1. 审核范围与 Git 基线

复核阶段 0、1A、对齐摘要、1B、1C、1D-0、1D-1 报告，课程任务书/报告模板/贡献声明及预测交接规范；检查现有模型、映射、数据 Provider、训练器、配置、指标、Checkpoint、测试、Git 历史和阶段证据。

开始时实际分支为 `dxc`，HEAD 为 **`663f727949a4b923fa84c3b76ec64efad7bd72a1`**；原先 13 个未跟踪文件全部存在，未发现已暂存或受跟踪文件修改。包含 D0 四文件和 D1 九文件，没有遗漏 D0。清单见 E/`initial_git.json`、`initial_files.json`。

origin 为 `git@github.com-dxc:dxc5731/NLPExp.git`，对应 GitHub 仓库 `https://github.com/dxc5731/NLPExp`。已正常 fetch 和实际 `ls-remote`：远端 dxc 与本地 HEAD 均为 `663f727…`，左右差异 **0/0**；不是由于分叉或 SSH 失败而停止同步。复用既有 dxc SSH 配置，没有读取或公开私钥内容。

本轮只修改 `05_character_adapter_integration.md` 的说明，新增本报告及 `stage1d_closeout_review.py` 审核工具。既有 140 个受跟踪文件、D0 四文件、v2 原样快照合计 163 个受保护文件通过哈希核对；原 D1 的实现/测试/配置/脚本保持不变，05 的文档修订为本轮明确授权变更，历史证据和源码快照保留原版本。

## 2. 本轮实际执行与结果

使用原 `.venv-bert-crf`，Python **3.13.5**、PyTorch **2.14.1+cu132**、Transformers **5.20.1**、tokenizers **0.23.3**、CUDA runtime **13.2**、RTX 5060 Laptop（CC 12.0）。未安装、升级依赖或新下载权重。

| 实际命令记录名 | 本轮结果 | 退出码 |
|---|---|---:|
| `readiness_probes` | 正式 Trainer/Provider 模式均被拒绝；未创建训练目录/优化器 | 0（拒绝被探针捕获） |
| `formal_entry_rejection` | 正式 CLI 对 `real_training` 配置抛出 RuntimeError | **1，预期的负向探针** |
| `remote_snapshot` | GitHub dxc 与本地基线相同 | 0 |
| `public_v2_tests` | **13 passed，0 failed**；unittest 3.413 秒 | 0 |
| `full_coverage` | 重新扫描完整内部 train/val，通过；进程 145.546 秒 | 0 |
| `real_cuda` | 重新执行真实小 Batch/八例/梯度/零更新/重载，通过；进程 14.515 秒 | 0 |
| `full_regression` | **188 passed，0 failed，0 skipped**；pytest 19.02 秒，进程 22.049 秒 | 0 |
| `dependency_check` | No broken requirements found | 0 |
| `file_and_evidence_review` | 逐文件/旧日志/敏感内容审查，结构化结果见 E | 0 |
| `review_closure` | 汇总复测与阻塞、核对现场；执行成功，准备度 FAIL | 见 commands.jsonl |

188 项包含原 130 项和新 58 项，没有省略真实权重/GPU集成参数，也没有通过跳过测试凑齐通过数。旧 `test_trainer_pretrained` 核对保留的阶段 1C 人工运行记录/Checkpoint，未重跑那次真实 BERT 人工优化；轻量人工训练器更新和 CPU/CUDA Epoch 恢复本轮实际执行。真实数据路径始终没有 optimizer.step。

关键命令如下；runner 设置 UTF-8、离线模式、正确公共接口路径、数据路径和 CUDA 确定性变量。完整运行覆盖和子进程源文件哈希都保存在 E/`commands.jsonl`。

```text
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d_closeout_review.py prepare
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d_closeout_review.py run readiness_probes -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d_closeout_review.py probes
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d_closeout_review.py run formal_entry_rejection -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 -m bert_crf.train --config .local/bert_crf/20261020T171714+0800_stage1d_closeout_review/negative_probe_real_training.json
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d_closeout_review.py run public_v2_tests -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 -m unittest discover -s .local/bert_crf/20261020T161159+0800_interface_v2_audit/interface/code/tests -v
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d_closeout_review.py run full_coverage -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d1_integration_check.py scan
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d_closeout_review.py run real_cuda -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d1_integration_check.py cuda
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d_closeout_review.py run full_regression -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 -m pytest tests/bert_crf -v -s --basetemp=.local/bert_crf/20261020T171714+0800_stage1d_closeout_review/pytest_all --pretrained-dir=.local/bert_crf/20261019T174356+0800_stage1b/pretrained --smoke-output=docs/bert_crf/evidence/20261020T171714+0800_stage1d_closeout_review/core_gpu_regression.json --checkpoint-dir=.local/bert_crf/20261020T171714+0800_stage1d_closeout_review/core_regression_checkpoint --trainer-smoke-summary=docs/bert_crf/evidence/20261019T192215+0800_stage1c/pretrained_trainer_final.json --junitxml=docs/bert_crf/evidence/20261020T171714+0800_stage1d_closeout_review/full_regression.xml
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d_closeout_review.py run review_closure -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1d_closeout_review.py closure
```

`negative_probe_real_training.json` 只用于验证入口拒绝，是本地负向测试输入，**不是已确定或已审核的首次训练配置**。没有启动正式训练进程后再中断，程序在构造优化器前已经拒绝。重跑须另建状态、日志、pytest 输出及 Checkpoint 路径，不能覆盖本次证据；审核 runner 对已有 state 和同名日志明确报错。

## 3. 数据、映射、模型与训练器结论

**重新全量实测：**train 2225 对话、88842 句、1153312 字符；val 247 对话、9687 句、126049 字符。共 **98529 句、1279361 字符**，每字符恰好覆盖一次；缺口、重叠、非法 offset、原字符 gold 长度不等、非法 BIO、拒绝样本全部为 **0**。原始 gold 完全一致句数为 98529。输入数据、接口 ZIP、split、Tokenizer、权重的当前字节哈希与 D1 基准一致，见 E/`input_identity.json`。

**静态与实测相互核对：**`V2DataAdapter` 依据组合 ID 安全读取原 `Dataset.char_labels`，对照原文/Token 编码，不使用 Token labels；固定 11 标签顺序不变。`offset_share_unique_v1` 的 gather 保持字符顺序和长度，覆盖冲突明确拒绝；人工已知向量/梯度累加、真实 GPU 链路以及 train/无标签 decode 同一逻辑通过。原 8 条歧义的定位来自 D0 实际文件；本轮再次核对原 gold 和文本哈希，全部有限 NLL、等长合法 BIO 解码，没有改金标。

**真实 GPU 复测：**train Batch `[2,10]` Token 输入、`[2,8]` 字符标签，NLL **11.773172378540039**，完成 backward。BERT 197/197、Linear 2/2、CRF 3/3 参数张量均有有限且非零梯度。val Batch `[2,40]` Token、`[2,39]` 字符，解码长度 `[39,5]`，NLL **53.44755554199219**；没有计算正式 F1 或选模。参数完整哈希和版本计数前后相同，优化器未创建，更新数 0。峰值 allocated **997216768 bytes（951.02 MiB）**，reserved **1082130432 bytes（1032 MiB）**。

集成模型保存/重载的参数哈希、eval 发射和路径相同；原 CRF 枚举、BIO 训练/解码硬约束、Mask/Padding/-100、人工训练器异常/恢复和日志测试全部通过。**但这些测试设计上仍验证人工训练或无更新真实集成，不覆盖尚不存在的正式训练启动路径。**

## 4. 证据完整性与公开范围

D0 的原始 ZIP/SHA、19 文件清单、v1/v2 diff、真实对齐/边界审计、演示和 13 项测试日志存在；D1 的覆盖 JSON、八例、真实 Batch、梯度、参数未变、模型重载、失败/修复日志、188 项 XML 和源码快照存在。原证据不覆盖、不清理。逐条旧命令日志哈希核对结果在 E/`file_and_evidence_review.json`。

D0 的 demo 原始日志含医疗原文，只保留于当时 `.local/.../logs/demo.log`；数据、权重、大 Checkpoint、环境、缓存及全部原始证据目录继续由原 `.gitignore` 排除。本轮不会强制添加任何忽略文件，没有为失败审核执行公开证据导出或上传。

对 13 个待审核 D0/D1 文件及本轮报告/审核脚本逐项检查文件类型/大小、医疗文本、密钥格式、用户目录绝对路径、项目路径和临时文件。辅助扫描只识别若干密钥格式及官方 train 中长度至少 8 的完整句精确匹配，**不能证明任意短片段或所有格式秘密都不存在**；同时结合源码和说明人工复核。新报告保留项目相对路径和必要复现目录，没有复制个人私钥路径、账号环境或原始医疗文本。`C:/userProgram/NLPexp` 只作为历史项目路径，不是可直接用于其他机器的默认环境；环境快照的完整本地路径保留在忽略目录。

未暂存任何文件，暂存区为空；`git diff --check` 检查受跟踪差异，未跟踪文件另按清单审查，不把空 diff 当作已经审查这些新文件。收尾清单保存全部候选源文件哈希和最终状态，供后续修复审核使用。

`review_closure.json` 是检查器结束时的快照：重新读取 XML、覆盖/八例/梯度/重载结果，验证既有日志哈希及 Git 现场，并记录全部 15 个未跟踪文件的当前哈希。逐文件审查之后只继续补充本报告和审核工具的收尾校验；两者的最终哈希以 closure 为准。closure 内的命令清单哈希取自本次 closure 命令记录追加之前，避免自引用；其实际退出码和日志哈希由外层 runner 随后追加到 `commands.jsonl`。审查完成不等于正式训练准备度通过。

## 5. 监督粒度与最终评估提醒

已实际修订 `05_character_adapter_integration.md` 的独立第 8 节 **“与 BERT-Softmax 的监督粒度差异及最终评估注意事项”**，并修正第 7 节将 Softmax 最终监督方式写成确定事实的措辞。

已核验的事实：本 CRF 使用原字符 gold，经 offsets 共享 Token 表示后在字符序列上建模，输出每原字符一标签；v2 随包提供的 Softmax 构造函数是 `AutoModelForTokenClassification`，公共 Dataset 的 Token labels 对歧义词片置 -100，文档建议交叉熵忽略。**Softmax 成员实际正式训练是否采用该路径未证实**，需以其代码、配置和日志核实。

最终评估应统一原始 ID/字符顺序/预测长度、11 标签映射、严格句内完整实体边界匹配、Micro/Macro 和零分母口径；记录各方法监督粒度、字符映射及异常/修复规则。若监督粒度不同，比较的是完整方法，不能把差异全部归因于 CRF。单独研究输出层贡献时，应由本人/团队设计尽量只改变输出层的控制实验，本轮不开展实验或填写研究结论。

## 6. 正式训练前置条件逐项审核

| 前置条件 | 判定与证据 |
|---|---|
| 原文、组合 ID、原始字符 gold 获取稳定 | **通过，真实全量/八例实测**；原 JSON、Dataset、Batch 同一身份 |
| 映射规则版本可配置、可追溯 | **通过**；`offset_share_unique_v1` 和 Unicode 坐标在配置、Provider 元数据、源码/证据中固定 |
| 训练 Loss 与无标签解码使用同一映射 | **集成层通过**；同一 `CharacterBertCRF`/Adapter，去 gold 后路径一致；正式训练入口未接通 |
| 独立正式训练配置 | **未满足，静态检查及配置枚举**；现有三个 JSON 仅为核心 Smoke、人工 Trainer Smoke、真实集成测试，无 `real_training` 配置 |
| train/val 边界 | **当前数据通过**；只读官方 train 内固定名单，互斥且数量一致；正式启动器尚未实现固定名单/哈希的接线与强制检查 |
| 最佳 Checkpoint 默认严格实体 Micro F1 | **已有框架通过人工测试**；`entity.micro.f1/max`，严格改善、平分保留早者；未在真实训练中选模（本轮禁止） |
| Loss/每轮指标/LR/种子/环境/Git/耗时自动记录 | **已有框架静态及人工回归通过**；Experiment/history/manifest；没有真实训练记录，不冒充已验证正式入口 |
| 异常、Checkpoint、Epoch 恢复 | **基础框架通过人工 CPU/CUDA 测试**；真实 Adapter 模型重载通过；真实 Provider 的完整训练状态恢复尚未接通/验证 |
| 8GB 资源控制 | **部分准备，未完成正式配置验收**；短 Batch 无更新峰值约 951 MiB，历史人工 AdamW Smoke 约 2 GiB；不能外推到最长真实句的联合优化。当前 max_len=256、num_workers=0、梯度裁剪及 OOM failed 记录可复用；正式 Batch/长序列内存预算未固定 |
| 人工模式不会误触发真实训练 | **拒绝保护通过**，但同时存在下述阻塞：真实模式也无可执行路径，不能把“任何真实训练都拒绝”当成正式入口已完成 |
| 最终预测交接纳入计划 | **文档计划通过，未实现最终导出/全平台验证**；05 第 8 节与交接规范明确保留，首次训练准备不能冒充最终推理已验收 |

因此“模型与真实字符数据已经可接”成立，“下一步只需授权即可第一次正式微调”不成立。

## 7. 实质性阻塞、影响与重新验收条件

**阻塞 R1：正式执行路径没有完成，代码事实并已复现。**

- `src/bert_crf/trainer.py:59` 调用 `require_synthetic(config)`；`training_config.py:74` 只接受人工类型，真实模式直接 RuntimeError。即使改配置也不能构造真实 Trainer。
- `src/bert_crf/train.py:12` 同样调用该保护，下一行还是无条件 RuntimeError。它不是“等待授权开关”，没有构造 Provider、模型或调用 Trainer 的分支。
- `src/bert_crf/data_adapter.py` 的 `V2BatchProvider` 仅接受 `real_data_integration_test`，拒绝正式训练类型。不能把真实数据标成人工以绕过。
- E/`readiness_probes.json` 记录前两类对象级拒绝；L/`logs/formal_entry_rejection.log` 保留真实 CLI 退出 1。报错中“Adapter 未集成”是旧提示，当前字符 Adapter 已验收；实际代码原因是执行模式约束及入口未实现。

这三个点属于同一项正式集成阻塞，需要一并解决；单独删除一个 require 或只改报错文案不足以验收。

**阻塞 R2：独立正式配置及其启动校验缺失。** 没有正式运行配置来固定数据/split/Tokenizer/映射身份、model/runtime、首次训练参数和资源边界。现有集成配置明确 `optimizer_steps=0`；本轮负向探针配置也不能当正式配置。首次实验参数需要本人确认，本轮不代替本人完成正式实验设计。

建议后续先完成范围明确的正式模式集成：保留人工/真实严格分离，显式构造已审计数据 Provider 和真实模型，校验唯一/互斥 ID 与输入版本、独立配置/输出目录和映射身份，连接已有 Trainer/Checkpoint/记录系统；以人工样例验证正式调度和恢复、以允许的无更新真实 Batch 验证配置接线及资源，不开展正式微调。再验收拒绝误用和正常路径、完整回归及元数据后，才重新做提交/同步审核。不要通过标签修复、更换数据策略或重写 CRF 解决入口问题。

**不单独视为本轮提交阻塞的后续事项：**一般空白/超长输入仍明确拒绝；Softmax 最终监督方式待核实；最终预测导出/统一校验工具待交接。它们影响后续通用推理或公平比较，已经记录，不应虚构为当前完整内部数据出现了覆盖错误。

本轮没有修复实质性训练代码，以免把审核扩大为新的模式/配置设计；只修正已充分确认的文档措辞，并保留完整失败现场。没有发生新的单测失败或实际 OOM。

## 8. 本轮文件、Git 决定与停止点

本轮实际修改：`docs/bert_crf/05_character_adapter_integration.md`。
本轮新增：`docs/bert_crf/06_pretraining_readiness_review.md`、`docs/bert_crf/stage1d_closeout_review.py`；以及被忽略的独立审核证据/日志/Checkpoint/负向探针输入。

原 13 个未跟踪成果完整保留，增加本轮两新文件后为 15 个未跟踪文件；没有受跟踪源码修改和暂存。收尾以 E/`review_closure.json` 的实际 Git 状态为准。

按本轮用户的通过条件和失败处理规则，**没有执行 git add、commit、push、历史改写或强制操作**。当前没有新的本地提交或 GitHub 提交链接，远端仅核实到已有基线；不能称本轮代码已经上传。后续修复通过后才能重新审核这些完整 D0/D1 成果并正常提交。

**最终审核不通过，尚不具备立即开始首次正式微调的完整工程条件。** 已验证的阶段 1D 数据/模型集成成果继续有效。本轮停止，等待本人决定后续正式入口和配置集成，不启动训练。

Codex 的参与范围为源码/材料审查、文档修订、审核工具、实际复测与工程阻塞定位；统计来自程序，正式研究动机、实验结论和贡献声明未代写。
