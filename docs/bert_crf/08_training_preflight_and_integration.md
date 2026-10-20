# 阶段 1E：正式训练流程接通与启动前验收

日期：2026-10-20，Asia/Shanghai。独立运行 `20261020T174150+0800_stage1e`。

**技术验收通过；正式真实 Epoch 训练未启动。** 阶段 1D 收尾报告的 R1/R2 已通过代码接通、独立配置、真实无更新 Preflight 和隔离训练测试解决。本人审核配置并明确授权后，可使用 `first_baseline.json` 开始第一组正式基准。参数效果与真实完整 Epoch 运行尚未验证，不把它们写成已知成果。

E：`docs/bert_crf/evidence/20261020T174150+0800_stage1e/`；L：`.local/bert_crf/20261020T174150+0800_stage1e/`；P：同编号的 `docs/bert_crf/published_evidence/` 子目录。完整敏感日志、数据和 Checkpoint 留在忽略目录；P 保存允许公开的轻量证据和转换清单。

## 1. 基线、范围与改动

开始时实际分支 `dxc`、HEAD `663f727949a4b923fa84c3b76ec64efad7bd72a1`。原 15 个未跟踪文件全部保留；阶段 1D-0 四文件、1D-1 九文件和上轮审核两文件没有遗漏。运行前将已有受跟踪/未跟踪文件全部保存哈希及本地快照，见 E/`initial_manifest.json`、`initial_git.json` 和 L/`initial_snapshot/`。

阅读了课程任务书、报告模板、贡献声明、交接规范和 00–06/对齐摘要，结合现有源码检查。`integrated_model.py` 在本项目中不存在；集成模型实际为既有 `model.py` 的 `CharacterBertCRF`，直接复用，没有新增另一套模型。

| 文件 | 工程改动 |
|---|---|
| `src/bert_crf/formal_config.py` | 独立完整 schema，拒绝缺失/未知/重复 JSON key，固定已审计数据与模型身份、标签和映射，校验实际 warmup 步数 |
| `src/bert_crf/real_data.py` | 哈希校验后隔离导入 A 的原样 v2 Dataset；仅官方 train 内固定名单；连接原字符 Adapter 和正式 train/val Provider |
| `src/bert_crf/data_adapter.py` | 最小扩展 Provider 模式，正式模式必须带完整已验证身份；保留原 integration 模式，字符金标和映射逻辑不改 |
| `src/bert_crf/preflight.py` | 全量字符/长度校验、输出/资源合同、最长真实 Batch 前后向和 val 解码；只产生无更新检查，不构造优化器 |
| `src/bert_crf/train.py`、`scripts/train_bert_crf.py` | 必须显式 `--preflight` 或 `--train`，互斥且无默认动作；正式模式先执行完整 Preflight，失败不得构造 Trainer |
| `src/bert_crf/trainer.py` | 人工保护保留；正式模式须绑定 PASS 上下文的同一模型、Provider、配置与输出根；复用原 Epoch/指标/早停/恢复流程 |
| `src/bert_crf/run_artifacts.py` | 每轮内部 val 真实解码 JSONL、严格实体错误跨度、最佳结果摘要和内部 run_info，部分输出不标完成 |
| `checkpoint.py`、`experiment.py`、`config.py` | 增加可选运行身份/完整配置；正式日志为 run.log；记录 CUDA 确定性环境；不改 CRF 算法 |
| `configs/bert_crf/first_baseline.json` | 首份独立正式配置；完整参数与输入哈希；资源测量的可定位文件和哈希 |
| `tests/bert_crf/test_formal_training.py` | 新增 51 项配置、Provider、启动保护、人工更新/记录/恢复、故障、真实 Preflight/资源证据测试 |
| `docs/bert_crf/stage1e_*.py`、07/08、README | 命令记录、实测、公开证据与收尾审查、复现指引 |

A 的公共源码、Tokenizer 文件和固定划分未修改。原 `crf.py`、`constraints.py`、`character_mapping.py`、`model.py` 算法原样保留。已有原始证据不覆盖；05 的监督差异第 8 节保持不变。06 是上轮实际失败记录，不将其改成历史成功；本报告记录之后的修复。

## 2. 正式路径与保护

正式配置 → 全部输入字节哈希 → A 的原样 Dataset/固定名单 → 复合 ID/原文/字符 gold 校验 → `V2BatchProvider`（train=`real_training`，val=`real_validation`）→ BERT → `offset_share_unique_v1` → 字符 CRF → 原 Trainer。

模型需要的所有字段经包装器传入，公共 Token labels（含 -100）不进入字符 CRF。样本缺失、错配、非法 BIO/Mask/offset/覆盖均明确拒绝，不跳过样本。train shuffle 使用专用 seed+epoch Generator；val 固定源顺序。全量审计确认 train/val 的对话和句子身份互斥、完整。

`Trainer` 没有删除 `require_synthetic`。人工入口仍拒绝真实 Provider；无 PASS `ReadyContext` 的真实 Trainer 仍拒绝。正式上下文绑定同一模型、Provider、正式配置 fingerprint 与输出根。CLI 只有显式 `--train` 分支能调用 `fit`；`--preflight` 从不创建优化器或正式实验 history。命令分派测试使用 spy，没有偷偷执行真实 fit。

Preflight 逐项输出 PASS/FAIL，保存配置、环境、Git/dirty 快照及代码哈希。校验公开资源证据的 hash、模型/输入版本、软件/GPU 一致性、最长实测覆盖、零更新证据和至少 1 GiB 余量；随后在当前 GPU 再测最长 Batch。输出根固定 `experiments/bert_crf`，每个 run 唯一，不覆盖旧目录；Preflight 目录不得写进输入、源码或正式实验目录。写盘/磁盘不足、配置或版本不符会拒绝。

CUDA 确定性按配置显式启用，TF32 关闭；运行时若未设置 CUBLAS_WORKSPACE_CONFIG 则采用 `:4096:8`，若已有不支持值则拒绝，实际值写入环境。所有有效训练参数保存，不提供未记录的环境变量覆盖、自动降精度/Batch 或人工回退。

## 3. 配置和真实资源依据

见 [逐参数决定依据](07_baseline_config_decisions.md)。A/B/C 明确区分：团队固定条件、本人候选值、Codex 实测推荐值；没有把候选值写成最优值。

当前建议：FP32，Batch 16，动态 Padding，上限 256 且拒绝截断；AdamW 全参数 LR=2e-5、weight_decay=0.01、foreach=False；5 Epoch、Early Stop patience=2；Linear warmup ratio=0.1（2777/27765 步）；seed=42；clip=1.0；dropout=0.1；accumulation=1、workers=0、无 AMP。

完整真实 Token 长度：train/val 都为 min=3、median=11、P90=30、P95=40、P99=68，max 分别 201/155。共 98529 句、1279361 原字符全部保留；全量覆盖与 gold 校验无拒绝。没有超过上限的句子。

资源测试实际 4/8/16 的普通和最长唯一样本六种组合均通过，无 OOM；每次额外预留约 1.14 GiB 的 AdamW 容量，Batch 16 最长 `[16,201]`/字符 `[16,199]` 峰值 allocated 3.598 GiB、reserved 3.818 GiB，仍可用约 2.97 GiB。所有参数哈希和版本计数不变，BERT/Linear/CRF 梯度有限且非零。

单步普通/最长约 0.240/0.649 秒，最终独立首次压力步约 1.155 秒，val 小 Batch 约 0.062 秒。计时边界、CPU 适配、Padding 开销和每 Epoch 粗略预算详见 07；完整真实 Epoch、真实优化器吞吐及正式成绩均未测量。不能从短句或单个时间点保证所有硬件状态不会 OOM，所以每次启动仍复查最长 Batch 和余量。

## 4. 本轮实际测试与命令

环境实际为原独立 Python 3.13.5、torch 2.14.1+cu132、Transformers 5.20.1、CUDA 13.2、RTX 5060 Laptop。未升级依赖、重新下载权重或使用官方 dev/test。

| 运行 | 真实结果 | 退出码 |
|---|---|---:|
| 候选配置生成 | 独立配置通过结构检查；资源绑定待测时不能 Preflight 成功 | 0 |
| 初次长度/覆盖 survey | 全量通过，使用候选 Batch 8 估算 Padding | 0 |
| 六组合资源测试 | 4/8/16 普通/最长，有限 Loss/梯度、哈希不变 | 0 |
| 资源配置选择 | 推荐最大实测安全候选 Batch 16，绑定测量 hash | 0 |
| 首批新增测试 | 42 passed，1 deselected（真实证据项留待 Preflight） | 0 |
| 初次真实 Preflight | 全量 train/val、最长前后向、val 解码 PASS，无更新 | 0 |
| 首次完整回归 | 234 passed，0 failed，0 skipped（随后增加保护测试） | 0 |
| 真实 BERT 人工 Trainer | 4 条人工文本、2 短 Epoch、4 optimizer 步，BERT/Linear/CRF 实际更新；最佳重载通过 | 0 |
| 最终真实 Preflight | **全部 PASS**；进程 75.226 秒，内部检查 68.414 秒 | 0 |
| 最终完整回归 | **239 passed，0 failed，0 skipped**；pytest 37.33 秒，进程 40.644 秒 | 0 |
| 公共 v2 测试 | **13/13**；unittest 5.561 秒 | 0 |
| pip check | No broken requirements found | 0 |

完整 argv、时间、退出码、源文件哈希和原日志 SHA 在 E/`commands.jsonl`，原日志在 L/`logs/`。最终配置 Preflight 命令实际执行了：

```text
.venv-bert-crf/Scripts/python.exe -X utf8 scripts/train_bert_crf.py --config configs/bert_crf/first_baseline.json --preflight --preflight-output docs/bert_crf/evidence/20261020T174150+0800_stage1e/preflight_final
```

完整回归通过 runner 设置审计路径和离线/确定性环境：

```text
.venv-bert-crf/Scripts/python.exe -X utf8 docs/bert_crf/stage1e_runner.py run full_regression_final -- C:/userProgram/NLPexp/.venv-bert-crf/Scripts/python.exe -X utf8 -m pytest tests/bert_crf -v -s --basetemp=.local/bert_crf/20261020T174150+0800_stage1e/pytest_all_final --pretrained-dir=.local/bert_crf/20261019T174356+0800_stage1b/pretrained --smoke-output=docs/bert_crf/evidence/20261020T174150+0800_stage1e/core_gpu_final.json --checkpoint-dir=.local/bert_crf/20261020T174150+0800_stage1e/core_checkpoint_final --trainer-smoke-summary=docs/bert_crf/evidence/20261019T192215+0800_stage1c/pretrained_trainer_final.json --junitxml=docs/bert_crf/evidence/20261020T174150+0800_stage1e/full_regression_final.xml
```

239=原 188+新增 51；原 130 核心/Trainer 和 D1 新增 58 项均包含，没有删测或放宽原断言。历史 Stage 1C 预训练测试仍明确核验旧记录；本轮另有新的真实 BERT 人工更新运行，二者不混为一次。旧 D1 的首次测试断言错误/修复记录保留在原证据和公开摘要，本轮没有出现新测试失败或真实 OOM，不制造迭代失败。

新增异常单测中故意注入的非法预测、Checkpoint 写失败和模拟 GPU 余量不足是标注清楚的单元测试，不是声称本机真实磁盘/GPU发生故障。runner 初次继承了上阶段 started_at 字段，已用真实 prepare 时间更正，并保留 `runner_metadata_correction.json`；命令自身时间与原日志未改写。

收尾工具首次 `acceptance_summary` 退出 1：将实验根目录必须为空作为断言，误拒绝原有两次 2026-10-19 的 `synthetic_smoke_test`。已读取原 manifest 核实并保留旧目录，将断言改为“没有新建或真实训练运行”，不是删除目录让检查通过。随后依赖缺失摘要的首次导出也退出 1；日志均保留，已停止该依赖链并在摘要成功后重新导出。部分已生成的轻量副本仅在与预期字节完全相同时复用，不覆盖不同内容。此为收尾脚本问题，未涉及模型/训练测试失败或真实参数更新。

## 5. 训练更新、验证和恢复的证据边界

**真实数据实测：**两个完整 Preflight 全量检查通过；最长 train Batch 有限 Loss（最终无更新探针 NLL 359.17974853515625）、反向和全部梯度通过；val Loss 32.000972747802734，输出等长且严格 BIO 合法，不计算正式 F1/选模。每个真实 probe 的前后全参数 hash 均为 `40a93116…68b088a`，版本计数相同、optimizer_steps=0。没有创建真实训练优化器，正式实验目录没有真实运行。

**人工更新实测：**新真实 BERT 人工运行 BERT embeddings、Linear、CRF 的最大变化分别约 4.01e-5、4.01e-5、3.92e-5；人工 history.csv 记录两轮有限 Loss，最佳模型重载验证通过。轻量模型的连续执行与 Epoch 边界恢复，model/AdamW/scheduler/RNG、步数和非耗时历史一致；CPU/CUDA 旧恢复测试本轮重跑通过。包含新 validation JSONL/最佳摘要/run_info 的人工恢复也通过。

**静态与隔离流程验证：**正式分支连接同一 Trainer、AdamW、Linear scheduler、完整 val、严格 Micro F1 选择、平分保留早者、早停、Checkpoint、事件记录和失败状态。正式真实完整 Epoch/真实完整恢复被本轮边界禁止，未执行，也未冒充已实测；工程路径由共享人工更新/恢复、真实无更新接口以及分派/保护测试共同验证。

Checkpoint 包含 model/optimizer/scheduler、Epoch/global_step、最佳指标/早停、Python/NumPy/Torch CPU/CUDA RNG、完整有效配置/来源运行 ID/代码身份；格式保留兼容的可选 metadata，不改旧必需状态键。只支持完成 Epoch 后恢复，恢复必须使用相同配置/Provider 身份及完整前运行目录（含最佳 Checkpoint 和 val 产物），不支持只挪走单个 Checkpoint 后悄悄补其他结果。恢复在新的 run 记录，不覆盖旧 history，不重新初始化 optimizer 后冒称恢复。混合精度关闭，amp_scaler=None 如实记录。

## 6. 正式实验的标准产物与交接

未来授权的真实运行目录自动产生：

```text
experiments/bert_crf/<unique_run_id>/
  config.json                  # training/model + 完整正式配置
  environment.json
  run_manifest.json            # 启动/结束、命令、cwd、Git/dirty/源码身份、状态、耗时
  source_manifest.json, source_snapshot/, tracked_changes.patch
  run.log, history.csv, checkpoint_info.json
  checkpoints/epoch_*.pt
  validation/epoch_*/predictions.jsonl, errors.jsonl, validation_info.json
  predictions.jsonl, errors.jsonl, metrics.json  # 最佳内部 val 结果，已验证来源哈希
  run_info.json                # bert_crf_internal_run_v1；非官方交接记录
```

预测保存原始 dialogue_id、sentence_id、字符长度和真实 BIO 解码，不写医疗原文；错误只列遗漏/多出实体的半开坐标及类别。实体必须同时匹配句子身份、边界和类别，零分母=0；字符 P/R/F1 排除 O，并另记固定五类 Micro/Macro。非法 BIO、重复 ID 或错长度明确失败，不用 gold 修预测。

每轮 val 在 no_grad/eval 下执行，计时包含 loader/适配、GPU 前向解码、指标和 JSONL 写盘，不含模型初次加载和参数更新；训练 fit 计时另记更新/val/日志/Checkpoint I/O，避免混用。run_info 区分最后一次验证和选中验证，记录真实时间、配置 hash、代码、数据/Tokenizer/映射身份和执行成员。失败保留 partial/failure 状态，不能进正式结果汇总；Checkpoint 写失败也不得标记 completed。

这些是**内部 val 训练产物**，不是 `predictions_dev.json`/`predictions_test.json` 或交接规范中 `artifact_kind=model_prediction` 的正式文件。本轮只在人工目录生成这些训练产物；真实 probe 不产生成绩和预测文件。未来官方推理仍需独立完成全平台覆盖、交接 JSON 与实际推理命令/耗时/哈希，未运行的字段不伪填。

05 第 8 节的监督差异提醒继续有效：CRF 原字符 gold、八例不损失监督、offset 共享；Softmax 最终监督方式待成员实际实现核实。最终评价须统一 ID/字符顺序、标签和严格边界规则；监督粒度不同时比较完整方法，不能把全部差异归因于 CRF。无其他组员代码改动或返工要求。

## 7. 公开证据、安全与 Git 收尾

审查包括初始未提交文件、当前差异、忽略目录、公共 v2/split 原样哈希、代码/测试/报告/配置和允许公开的轻量摘要；不无差别 add、不强加忽略文件。公开内容不含原医疗文本、完整敏感 demo 日志、模型参数、Checkpoint、数据 ZIP、环境目录或密钥；本地绝对项目路径仅历史复现定位，个人用户目录/主机名从新增公开摘要中移除。

公开证据是显式白名单转换，保留原源文件 hash、转换说明及日志 hash，原日志仍在本地。源码的最终清单和隐私扫描结果见 E/P 的审核清单；扫描只作为人工审查补充，不把若干正则声称为任意隐私的完全证明。

本报告所在提交整合未提交的 D0/D1 及本轮 E 成果，Git 基线为上述 663f727；只有全部审核通过后创建提交并正常 push `origin/dxc`。提交前再次 fetch/核对无分叉，提交/推送后校验 SHA 与工作区，实际最终 SHA/链接写入交付反馈及本地 `git_sync_verification.json`，避免在提交内部制造自引用 SHA。可用 `git log -1 --format=fuller -- docs/bert_crf/08_training_preflight_and_integration.md` 定位提交。若同步失败，则最终反馈不得称远端已完成。

## 8. 验收判定与停止点

正式 Provider、独立配置、显式入口/保护、数据/字符映射身份、全量长度统计、资源方案、人工训练/更新/调度/恢复/产物、真实无更新 Preflight、188 项回归及新增 51 项均通过。未发现尚待解决的实质性代码或数据阻塞。

配置值是已验证可运行的**首次基准建议**，本人仍需确认实验选择与启动授权；GPU占用变化会触发新的 Preflight，不是本轮尚未解决的问题。未来未知的正式 F1、Epoch耗时和模型效果必须由真实训练产生。

Codex 参与接口接线、配置校验/资源测量、单测与调试、文档/证据和 Git 工程操作；本人给定的初始候选值与团队条件单独列明。没有代写研究动机、正式实验结论或个人贡献声明。**本轮停止于验收、提交和同步，不调用真实 --train。**
