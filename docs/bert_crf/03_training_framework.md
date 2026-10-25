# 阶段 1C：训练框架、验证与可复现记录

本阶段类型为 `synthetic_smoke_test` / `unit_test` / `environment_check`。2026-10-19 实际执行；全部优化器更新、Loss 和指标均来自人工样例。未读取 IMCS 原文或金标，未进行正式微调、真实数据评估或推理，未实现真实字符映射。

## 1. 实现与边界

复用阶段 1B 的 `CharacterBertCRF`、BERT 编码器、字符 Head 和严格约束 CRF；`model.py`、`crf.py`、`constraints.py`、`config.py` 与执行前提交 `2706f86` 内容一致。公共接口 16 个受审文件逐一 SHA-256 相同，含固定划分文件。核对见发布证据的 `verification.json`。

新增代码：

| 文件（`src/bert_crf/`） | 功能 |
|---|---|
| `training_config.py` | 参数、运行类型及合法性校验；不允许正式训练 |
| `trainer.py` | Epoch/Batch、AdamW、Warmup、constant/linear 调度、梯度裁剪、验证、选择最佳、早停和异常记录 |
| `metrics.py` | 逐字符与严格实体指标，保留句子 ID，拒绝非法 BIO |
| `checkpoint.py` | 模型、优化器、调度器、训练计数与 RNG 的 Epoch 边界恢复 |
| `experiment.py` | 独立运行目录、配置、环境、代码快照及哈希、JSONL 日志、CSV 历史、状态 |
| `synthetic_smoke.py` | 专用人工 Provider、轻量模型测试与真实 BERT 最小训练器调用 |
| `train.py` | 正式入口占位；明确拒绝执行，不回退到人工映射 |

新增 `test_metrics.py`、`test_trainer.py`、`test_trainer_pretrained.py`；`conftest.py` 仅增加人工训练集成结果参数。其他新增文件为阶段配置、证据运行/核对脚本、空白实验记录模板和本报告。`.gitignore` 新排除 `/experiments/bert_crf/`。未修改其他成员模块。

数据流为：人工 BERT 输入 → 既有编码器 → **仅测试使用的一对一 gather** → 既有字符 CRF Head → Trainer；Head 接收 `[B,C,H]` 表示、`[B,C]` long 标签及 bool 连续前缀 Mask。训练器不接受 `-100`，Padding 使用合法占位标签并被 Mask 排除。真实 Adapter 仍为未实现契约。

模型维持 101,685,658 个可训练参数。使用 PyTorch 原生 AdamW/LambdaLR，未新增依赖或引入另一套 CRF。运行环境：Windows 11、Python 3.13.5、PyTorch 2.14.1+cu132、Transformers 5.20.1、RTX 5060 Laptop；完整当前版本在发布 `environment.json`。`pip check` 返回 `No broken requirements found.`，退出码 0。

## 2. 指标与选择口径

- 字符 Micro P/R/F1：按有效原始字符精确比较 BIO 标签，O 不参与 P/R/F1；B/I 不同也算错误。分类别指标聚合该类型 B/I 的计数，TP 仍要求 BIO 精确一致。另记包含 O 的字符准确率。
- 字符和实体 Macro：固定五种实体类型逐类计算后等权平均，包含零支持类别；所有零分母取 0。
- 实体：完整 `(dialogue_id, sentence_id, start, end_exclusive, type)` 严格匹配；不接受重叠代替完整匹配，不跨句合并。重复 ID、长度不等及非法 BIO 报错。
- 默认最佳指标 `entity.micro.f1`、越大越好、`min_delta=0`；仅严格改善才更换，平分保留较早 Epoch。支持配置方向及早停 patience；连续不改善达到 patience 时停止。缺失、非法或非有限选择指标报错。
- 训练/验证 Loss 为按样本数加权的序列 NLL 均值，不冒充逐字符平均。每个 Epoch 的 Micro 和分类别完整结果在 `train.log`，`metrics.json` 为最后一轮验证，不能当作最佳轮；最佳轮由 `checkpoint_info.json` 定位。

这些是本模块内部验证准备，未替代 D 的统一最终评估；正式指标口径需与评估负责人核对。

## 3. 实际测试结果

证据运行 ID：`20261019T192215+0800_stage1c`。原件在 `docs/bert_crf/evidence/<ID>/`，24 份已核对轻量副本在 `docs/bert_crf/published_evidence/<ID>/`；发布清单逐文件保留来源路径、字节数和 SHA-256。没有将权重、医疗原文或 Checkpoint 导出。

| 实际命令记录名 | 结果 | 退出码 |
|---|---|---|
| `first_framework_tests` | 初版框架测试 70 passed，11.27 秒 | 0 |
| `pretrained_trainer_smoke` | 首次真实预训练模型人工训练器测试 PASS | 0 |
| `pretrained_trainer_final` | 加入静态检查发现的保护措施后复核 PASS | 0 |
| `full_regression` | **130 passed，0 failed，0 skipped，14.98 秒** | 0 |
| `dependency_check` | 无损坏依赖 | 0 |

完整回归包含原 53 项核心测试和新增 77 项。涵盖已知答案字符/实体指标、边界/类型错误、零分母、BIO、Padding、优化器与调度器、早停/平分规则、状态恢复、目录及日志故障、配置/模型/Provider 不匹配、正式执行拒绝等。

本轮没有实际测试失败或实际 OOM。NaN/Inf、异常梯度、OOM、损坏 Checkpoint、日志不可写等为**显式故障注入的单元测试**，验证拒绝及 failed 状态，不能描述为真实训练遇到的失败。静态复查后增加了缺失运行类型、Checkpoint 状态结构、日志写入失败保护及相应测试；未虚构修复故事。Git CRLF 提示不属于测试失败。

提交前 `git diff --cached --check` 另发现 `test_metrics.py` 尾部多余空行（退出 2），只删除该空行后 19 项指标测试再次通过；不是算法修复。原始失败、复查及最终源码哈希另存发布目录的 `supplement/`，不覆盖上述 24 份原件。完整 130 项运行时与最终测试文件的唯一区别是此空白；运行模块未改变。

### 真实预训练 BERT 接入（只用人工样例）

固定来源 `google-bert/bert-base-chinese`，revision `8f23c25b06e129b6c986331a13d8d025a92cf0ea`。复用阶段 1B 的 `.local/bert_crf/20261019T174356+0800_stage1b/pretrained/` Tokenizer/config/权重，无新下载；`model.safetensors` SHA-256 为 `3404a1ffd8da507042e8161013ba2a4fc49858b4e3f8fbf5ce5724f94883aec3`。

验收运行：`experiments/bert_crf/20261019T192609+0800_synthetic_smoke_test_61102491/`。4 条人工训练和 4 条人工验证、每集合 10 个有效字符；两组使用同一批人工输入、不同 ID，专用于系统检查，**不是独立留出实验**。实际 Tokenizer 已核验人工输入一字一 Token 后才使用测试 gather。

配置 `configs/bert_crf/stage1c_trainer_smoke.json`：3 Epoch、Batch 2、lr 2e-5、weight decay .01、warmup 1、linear、clip 1、seed 1729、CUDA float32。6 次 optimizer 调用中首次 warmup 的 LR 为 0，其余 5 次非零。不是参数搜索。

| 人工测试 Epoch | Train NLL | Val NLL |
|---|---:|---:|
| 1 | 4.593286752700806 | 3.7128725051879883 |
| 2 | 3.8869065046310425 | 2.8646585941314697 |
| 3 | 2.9687983989715576 | 2.494633138179779 |

人工验证严格实体 Micro F1 依次为 .4、.3333333333333333、.3333333333333333，选择第 1 轮；只是验证选择规则，不能进入正式效果表。最佳重载后验证值与第 1 轮一致。BERT embedding、分类层、CRF 转移参数最大绝对变化分别为 `5.9232115745544434e-05`、`5.7635828852653503e-05`、`5.534570664167404e-05`，证明联合更新已实际发生。

fit 耗时 8.6005672 秒（包括更新、验证、日志与 Checkpoint I/O，不含初始模型/Provider 加载）；测试函数含加载 14.3718152 秒；外层进程 20.2804563 秒。峰值 allocated 2,104,820,736 bytes（约 2007 MiB），reserved 2,298,478,592 bytes（2192 MiB）。这些不是纯 GPU 推理计时。详细数字见原始 JSON/CSV/manifest。

## 4. Checkpoint 与复现验收

逐 Epoch 保存不覆盖的 `checkpoints/epoch_NNNN.pt`；`checkpoint_info.json` 指向 latest/best 并记 SHA-256，不复制另一份大文件。保存模型、AdamW、调度器、epoch/global step、best/bad epochs、配置及结构/Provider/版本标识、Python/NumPy/CPU/all-CUDA RNG；未启用 AMP，scaler 为 None。`load_best()` 仅加载模型用于验证并置 eval；恢复训练必须走完整状态加载。

在轻量随机 BERT 的 **CPU 和 CUDA** 上实际对照：连续 3 Epoch 与第 1 轮中断后恢复至第 3 轮，模型、AdamW、调度器、RNG 状态逐位一致；下一随机值、global step=6、最佳轮及历史数值一致（计时除外）。`resume_verification.json` 重新读取实际 Checkpoint 验证，含三段原始 manifest、历史和日志；pytest 还校验了恢复后的随机数值。

真实预训练大模型已实测更新及最佳重载，**没有实测完整大模型断点续训的逐位一致性**。不承诺跨硬件、CUDA、依赖版本逐位相同。仅支持 Epoch 边界，要求 Provider 无隐含游标/worker 随机状态，按 Epoch 独立播种；尚无多 worker、AMP、Batch 中途恢复。恢复须使用相同计划总轮数和配置（不支持直接改总轮数延长调度器）。最佳文件为本地绝对路径，移动运行目录后的重定位尚未实现。

真实 BERT 每份完整训练状态约 1.22 GB，保留本地未提交。最佳 Checkpoint 路径为上述运行的 `checkpoints/epoch_0001.pt`，SHA-256 `f775754eec5dcfc5bfc64c50c8b0f227975bd606023f5d45262ac43e20ae595b`。轻量发布副本只提供定位和哈希，不能单凭副本恢复模型。

## 5. 记录系统与重跑

每次创建时间/类型/UUID 唯一目录，已有目录拒绝覆盖；恢复建立新目录并记录来源 Checkpoint。保存 config、run_manifest、environment、source_snapshot/source_manifest、tracked_changes.patch、JSONL train.log、history.csv、metrics、checkpoint_info。运行状态区分 interrupted/failed/completed/early_stopped。真实结果汇总器仅接纳已完成的 `real_training`，排除人工记录。

所有实际命令、cwd、环境变量、时间、退出码和输出哈希在发布 `commands.jsonl`；源代码运行基于 `2706f861d29a72f40e7db9ffedbf56db8073a203` + dirty 工作区，保存源码快照/哈希，未回填已提交状态。最终提交可用 `git log -1 -- docs/bert_crf/03_training_framework.md` 定位；Git 提交发生于测试之后。

根目录 PowerShell 示例（命令无中文管道；每次使用新证据路径）：

```powershell
$env:PYTHONPATH = 'src'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.venv-bert-crf/Scripts/python.exe -X utf8 -m bert_crf.synthetic_smoke --config configs/bert_crf/stage1c_trainer_smoke.json --pretrained-dir .local/bert_crf/20261019T174356+0800_stage1b/pretrained --output-root experiments/bert_crf --summary .local/bert_crf/NEW_UNIQUE_summary.json
.venv-bert-crf/Scripts/python.exe -X utf8 -m pytest tests/bert_crf -v -s --basetemp=.local/bert_crf/NEW_UNIQUE_tests --pretrained-dir=.local/bert_crf/20261019T174356+0800_stage1b/pretrained --smoke-output=.local/bert_crf/NEW_UNIQUE_gpu.json --checkpoint-dir=.local/bert_crf/NEW_UNIQUE_checkpoint --trainer-smoke-summary=.local/bert_crf/NEW_UNIQUE_summary.json --junitxml=.local/bert_crf/NEW_UNIQUE_tests.xml
```

在本机该环境预期 130 passed；省略显式权重目录和人工训练 summary 时会跳过两项集成测试，不能声称全量验收。`stage1c_runner.py` 为本次执行包装器（实际 argv 和 UTF-8/offline 环境），`stage1c_finalize.py` 只核对/导出证据，不训练，拒绝覆盖导出目录。导出验证命令退出 0。`.gitignore` 排除环境、数据、Checkpoint、原始证据及实验目录；发布目录有逐文件清单及保留字节的 Git 属性。

`experiment_record_template.md` 仅提供空白记录字段，不填正式动机、成绩或结论。不生成假的未来 `run_info.json`。

## 6. 等待 A 的真实接口与下一轮

`BatchProvider` 保留 `size/metadata/batches(epoch,batch_size,shuffle,seed)`：每批 `ids=[(dialogue_id,sentence_id), ...]` 和 `inputs`。inputs 向现有模型传 BERT 输入、未来映射所需元信息、**原始字符金标**与连续前缀字符 Mask。批次必须完整覆盖原句集合，保留原始 ID，不能借 Token 标签替换原字符金标。metadata 预留数据/划分哈希、Tokenizer revision、标签映射与字符对齐策略版本；本轮仅填写人工版本。

A 后续需提供：不改原文的字符序列与字符 BIO、ID、原始字符长度、Token 输入及 offset/特殊 Token Mask、多字符 Token/未覆盖字符/长句的明确处理和覆盖审计、固定划分及来源哈希。真实 Adapter 应输出可微 `[B,C,H]` 与有效原字符一一对应；不得静默补 O、改 BIO 或丢尾部。具体空白/Unicode 表示与长句分段方案仍待接口交付后人工确认，本轮没有做通用适配。

正式入口 `python -m bert_crf.train` 明确拒绝执行；Trainer 本身也拒绝非人工类型及未标注/类型不匹配的 Provider。因此已有训练基础，但尚不具备正式真实数据训练条件。

下一轮最小任务建议：接收并只读核验 A 修复版，在原始字符金标不变的条件下测试覆盖/边界/Unicode/长句，单独实现真实 Adapter，再做极小真实 Batch 的前后向与验证协议核对。正式训练配置、首轮实验启动、长句/未覆盖字符策略需本人确认后再开展。

## 7. AI 使用与阶段结论

Codex 辅助实现代码、人工测试、调试防护和工程记录；程序实际执行了参数更新、指标、保存/重载、断点续训和回归核验。报告数字来自保留日志，没有代写正式实验动机、研究结论或个人贡献声明。本人仍需审核正式指标口径、A 的接口契约和后续实验决策。

本阶段 1C 工程验收已通过；真实 Adapter、真实数据训练/评估和完整大模型恢复一致性属于未验证范围。停止于本阶段。
