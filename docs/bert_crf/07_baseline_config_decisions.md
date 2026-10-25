# 首次 BERT-CRF 基准配置决定依据（阶段 1E）

2026-10-20，Asia/Shanghai。本文件是启动前工程配置记录，不是正式实验设计结论或医疗模型效果分析。真实训练尚未开始。本人仍须审核本配置并明确授权启动。

配置：`configs/bert_crf/first_baseline.json`，严格 schema `bert_crf_formal_v1`。实测运行：`20261020T174150+0800_stage1e`。下文 E 表示 `docs/bert_crf/evidence/20261020T174150+0800_stage1e/`；P 表示同编号的 `docs/bert_crf/published_evidence/` 子目录。公开文件是原证据的明确摘录或字节一致副本，转换关系见导出清单。

## 1. 决定来源与参数口径

A 为本人/团队在任务中已确定的条件；B 为本人本轮给出的初始候选值，工程上验证可用后采用；C 为程序实测后由 Codex 推荐的资源值。B/C 均不是已经证明的最优参数，后续正式对比须另建配置和独立实验目录。本轮不开展参数效果比较。

| 参数 | 类型、原候选与依据 | 当前建议值 | 后续是否调参/确认 |
|---|---|---|---|
| 数据与固定划分 | A；仅官方 train JSON，固定 split_v1 字节哈希与 1D 一致 | train 2225 对话/88842 句；val 247/9687，不重新划分 | 固定，不用官方 dev/test 选模 |
| 字符监督/标签表 | A；原 `Dataset.char_labels`，按组合 ID/原文校验 | 固定 11 BIO 标签顺序；Padding O 占位、mask 排除 | 固定，不从 Token labels 反推 |
| 字符映射 | A；全量唯一覆盖、八例保真和梯度证据 | `offset_share_unique_v1`；Python Unicode 码点；缺口/重叠/截断拒绝 | 本基准固定 |
| BERT | A/B；复用已审计权重与 Tokenizer 文件哈希 | `google-bert/bert-base-chinese`，revision `8f23c25b06e129b6c986331a13d8d025a92cf0ea` | 本基准固定 |
| 优化器 | B；已有 AdamW 及恢复测试，避免 foreach 内存临时列表 | `torch.optim.AdamW`，foreach=False，betas=(0.9,0.999)，eps=1e-8 | 起点；未做优化器比较 |
| BERT/Linear/CRF 学习率 | B；本人候选，当前单参数组真实行为一致 | 全部 `2e-5` | 本人审核；不是最优值 |
| Weight decay | B；本人候选；明确作用于整个参数组（含 bias/LayerNorm/CRF） | `0.01` | 本人审核；未做豁免组实验 |
| 最大 Epoch | B；本人候选，调度总步数基于上限 | 5 | 本人审核 |
| Early Stopping | A/B；严格改善才更新，平分保留较早者 | patience=2，min_delta=0，指标 `entity.micro.f1/max` | 本基准固定口径 |
| 调度/Warmup | B；Linear，warmup_ratio=0.1 | 5553 步/Epoch，最多 27765 步；ceil(27765×0.1)=2777 warmup 步 | 本人审核；Early Stop 不重新拉长调度 |
| 梯度裁剪 | B；既有有限梯度与裁剪测试 | norm=1.0 | 本人审核 |
| 随机性 | A/B；本人候选 42；Provider 独立 epoch RNG 不消耗模型 RNG | Python/NumPy/Torch CPU/CUDA seed=42，train shuffle seed=42+epoch | 固定并记录；不承诺跨平台逐位一致 |
| Dropout | B；沿用已验收模型值，配置完整保存 | 0.1 | 本人审核；本轮未比较 |
| Batch Size | C；真实 4/8/16 普通及最长样本压力测试 | **16**，最后不足 Batch 不丢弃 | 实测推荐；本人审核后首次使用 |
| Padding/max length | C；完整长度统计最大 Token 数 201/155 | 动态补到 Batch 最长；最大 256（含 CLS/SEP），truncation=False，超长报错 | 当前全量覆盖；新数据须重新审计 |
| 梯度累积 | C；显存余量足够，不引入新缩放/调度复杂度 | 1；其他值明确拒绝 | 当前不需要 |
| 精度/AMP | B/C；本人默认 FP32，最长 Batch 有余量 | float32，AMP=False，TF32=False | 不需要为了容量改精度；未做 AMP 实验 |
| DataLoader worker | C；复用已验证的 epoch-pure Provider，实测适配耗时较小 | 0，独立 Generator，不留 sampler 游标 | 未做多 worker 比较，不宣称速度最优 |
| 资源保护 | C；预留两份 AdamW moments + 一份参数工作空间 | 三份参数大小预留；至少 1 GiB GPU 余量、16 GiB 可用磁盘 | 每次 Preflight 动态复核，环境变化须重测 |
| 输出与格式 | A/工程实现 | 唯一 `experiments/bert_crf/<run_id>/`；内部字符 JSONL、严格实体指标 | 官方预测交接仍另行执行 |

首步 Warmup 学习率为 0 是记录的 Linear schedule 语义；scheduler 随每次 optimizer.step 递增，不按字符/Batch 内样本数递增。CRF Loss 为每句 NLL 的 Batch 均值，统计 Epoch Loss 时按实际样本数加权；不采用梯度累积。

## 2. 真实 Token 长度与 Padding

基于真实 Tokenizer 对整个内部 train/val 重新编码与完整字符校验，包含 CLS/SEP，不读取官方 dev/test。最终 `preflight_final/lengths_and_coverage.json` 为选定 Batch 16 的证据；最早 survey 使用候选 Batch 8，其 Padding 估算没有覆盖或伪装为最终值。

| 集合 | 样本 | min | median | P90 | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 88842 | 3 | 11 | 30 | 40 | 68 | 201 |
| val | 9687 | 3 | 11 | 30 | 40 | 68 | 155 |

分位数为 NumPy linear 插值；整数直方图和区间计数保留在 JSON。超过 256 或 BERT 512 上限的样本均为 0。每个原始字符唯一覆盖、金标长度和合法 BIO 全量验证通过，拒绝样本为 0，没有通过截断、修改标签或跳过样本达成。

Batch 16：train 按实际第 1 Epoch 的 seed=43 随机顺序，动态 Padding 共 4,246,304 Token 位置，为固定 256 的 22,743,552 位置的 **18.67%**；val 不 shuffle，按源顺序为 401,854 / 2,479,872，**16.20%**。这是张量位置数量的估计，不是等比例运行时间承诺，也不是改变数据顺序以优化指标。后续不同 Epoch 顺序的 Padding 开销可能不同。

## 3. 无更新真实资源测试

使用真实内部 train，普通样本来自专用 Generator seed=43，最长样本按 Token 长度降序取互不重复的前 B 条；不是重复复制同一条长句。每次清空梯度、释放上一图/预留、empty_cache、reset_peak、同步 CUDA；检查所有参数哈希与版本计数不变。没有创建真实数据优化器，没有调用 optimizer.step。

所有下表峰值都**额外包含 1,220,227,896 bytes（约 1.14 GiB）**容量预留，表示两份 FP32 AdamW 状态及保守工作空间；这是容量模拟，不是假称已经执行真实 AdamW 更新。

| Batch | 采样 | Token/字符形状 | peak allocated GiB | peak reserved GiB | 步耗时秒 |
|---:|---|---|---:|---:|---:|
| 4 | 普通 | [4,15]/[4,13] | 2.066 | 2.170 | 0.161 |
| 4 | 最长 | [4,201]/[4,199] | 2.072 | 2.260 | 0.353 |
| 8 | 普通 | [8,15]/[8,13] | 2.067 | 2.178 | 0.135 |
| 8 | 最长 | [8,201]/[8,199] | 2.596 | 2.791 | 0.436 |
| 16 | 普通 | [16,67]/[16,65] | 2.239 | 2.455 | 0.240 |
| 16 | 最长 | [16,201]/[16,199] | 3.598 | 3.818 | 0.649 |

Batch 16 最长时设备可用 3,192,913,920 bytes（约 2.97 GiB），高于 1 GiB 保护线。没有 OOM。全参数哈希均为相同前后值，BERT 197、Linear 2、CRF 3 个参数张量都有有限且非零梯度。随后独立最终 Preflight 的最长 Batch 峰值相同，首次前后向耗时 1.155 秒；普通 val Batch 推理/解码 0.062 秒。差异表明冷启动和运行状态影响计时，不能把单次测量当稳定吞吐量。

步耗时范围：预留分配、GPU 前后向或解码及梯度检查；不含分词、模型参数哈希、optimizer.step、Checkpoint 和日志 I/O。3 次 CPU 适配均值在所测 Batch 16 普通/最长分别约 0.011/0.027 秒；当前没有需要增加 Windows worker 的容量或明显吞吐证据。未开展 worker 搜索。

## 4. 时间预算与尚未测量事项

最多 5553 次更新/Epoch，val 606 Batch。用资源测试中 Batch 16 普通/压力的 0.240–0.649 秒机械外推，仅前后向约 **22–60 分钟/Epoch**；用首次 Preflight 最长步计时机械外推可接近 107 分钟。两者都不是真实 Epoch 运行结果，也不是正式训练成绩。

首次运行可预留约 **30–120 分钟/Epoch**的工程时间预算；这是包含未知更新/I/O/温控影响的粗略安排，完整耗时必须由获授权的首次训练记录修正。本轮没有测量全量真实优化器更新、完整 val F1、Checkpoint 大小下的每轮完整延迟或五轮实际耗时。人工 BERT 训练验证了 AdamW 和保存加载可执行，不能替代真实 Epoch 的性能测量。

当前没有资源证据要求梯度累积或混合精度，因此不增加这两条未经需要的实现路径。GPU 被其他程序占用、版本/硬件变化、数据/最大长度变化或改动模型时，Preflight 会因合同不匹配或余量不足拒绝，须重新测量，不能静默缩 Batch、降精度或截断。

## 5. 本人待审核项目

技术校验已给出一份可用的初始建议。本人仍应确认是否采用 B 类候选和 C 类推荐值、实验命名及首轮启动；这属于启动授权和实验选择，不是剩余代码阻塞。本轮不代填研究动机、预期效果、正式结果分析或个人贡献声明。后续不得因内部 val 分数不理想而无记录改参数；每次对比另留真实配置和独立运行证据。
