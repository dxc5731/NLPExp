# 预测交付与评价协议

原始数据保留 `data[dialogue_id]["dialogue"]` 数组，每句有 `sentence_id`、`sentence`，有标签集合另有 `BIO_label`。预测为 `predictions[dialogue_id][sentence_id] = "O B-Drug I-Drug ..."`。公开例子全部位于 `examples/`，均为人工构造。

每个原字符对应一个标签，单位是 Python `len(str)` 的 Unicode 码点，标点和空格也计数；不做文本规范化，不拼接 speaker，不混入 CLS/SEP/PAD。五类为 Symptom、Drug、Drug_Category、Medical_Examination、Operation，各有 B/I，另有 O，共11标签。标签以单个 ASCII 空格分隔，区分大小写。

校验先拒绝重复 JSON key、非 UTF-8/BOM、缺失或额外 ID、长度不一致、非法标签、句首 I 和跨类型 I。I-X 只可接同句 B-X/I-X；不补 O、不修复预测。PASS/FAIL/NOT_RUN 分别对应退出码0/1/2；缺原文的空 test 模板不能证明字符对齐，返回 NOT_RUN。

实体由 `(dialogue_id, sentence_id, type, start, end)` 唯一标识，坐标为原始字符零起点半开区间 `[start,end)`。边界与类型全部一致才是 TP，其余预测为 FP、金标为 FN。先全局汇总计数，再算 `P=TP/(TP+FP)`、`R=TP/(TP+FN)`、`F1=2TP/(2TP+FP+FN)`；零分母记0。不做部分匹配，不取句均或五类平均充当主指标。

字符 Accuracy 包含 O；非 O micro 仅排除 O 这个统计标签，保留全部原位置，仍统计 O→实体的 FP 与实体→O 的 FN。macro 固定使用11或10标签，缺支持类别 F1=0。混淆矩阵行为金标、列为预测。作者历史脚本中名为 token-level 的 seqeval 实体统计不作为真正字符指标。

评分复用校验时已加载并计算哈希的输入快照，避免校验后重读变化文件。输入不合格则仅写拒绝记录；官方 test 无公开金标，校验通过后仍拒绝本地评分。批量比较核对集合、金标哈希、指标策略、run 身份及元数据，合成与真实结果不得混入同表。

错误导出只把互为唯一的一对 FP/FN 关联为边界或类型差异，重叠歧义保持未配对。这是定位现象，不是自动归因。即使不导出原文，错误记录也含真实 ID 和坐标，真实数据运行的全部输出必须留在受控本地；`--local-text` 还会包含原文，禁止收入发布内容。
