# 天池与GitHub数据来源核查

核查日期：2026-10-25。本文核对作者两个仓库、CBLUE官方说明、本地原始平台ZIP及既有下载记录；train/dev差异由本轮只读比对重新计算。聚合结果见 [source_comparison_summary.json](source_comparison_summary.json)。

## 结论与实际差异

GitHub原研究仓库 **确实公开 `dataset/`**，包含 `train.json`、`dev.json`、`test.json`、`test_input.json` 和症状词表。它与天池包具有相同的train/dev对话规模和原始句子，但本轮固定版本比对发现标注和字段差异，因此全组采用同一份天池正式ZIP作为实验数据基准。

|核查项目|作者原研究仓库 `imcs21`|平台包 `IMCS-V2-NER.zip`|实际核查结果|
|---|---|---|---|
|用途与入口|论文基准的数据、基线和实验代码；README将2.0及平台评测指向 `imcs21-cblue`|CBLUE的IMCS-V2-NER任务数据与提交配套文件|两者有共同来源，平台包直接对应本项目所提交的任务|
|训练/验证文件名|`dataset/train.json`、`dataset/dev.json`|`IMCS-V2_train.json`、`IMCS-V2_dev.json`|文件名与发布包装不同|
|train规模|2472段对话、98529句|2472段对话、98529句|全部对话/句子ID对齐，句子文本差异为0|
|dev规模|833段对话、33267句|833段对话、33267句|全部对话/句子ID对齐，句子文本差异为0|
|NER标注|五类实体、11种BIO标签|相同五类实体及11种BIO标签|标签集合相同，但train中17653句、dev中5845句的BIO序列不同；分别涉及70684、22007个字符标签|
|句内其他标注|含 `symptom_norm`、`symptom_type`|同名字段，另有 `local_implicit_info`|train/dev的归一化症状字段分别有10335/3138句不同，症状类型字段分别有14096/4227句不同；平台版每句新增局部症状信息字段|
|test与交付文件|目录同时提供 `test.json` 与 `test_input.json`；README将前者说明为带训练/验证同类格式的测试数据|`IMCS-V2_test.json` 为811段、32935句的无标签输入；`IMCS-V2-NER_test.json` 为对应ID的空提交模板|本轮仅核对GitHub目录与说明，未打开其测试答案；平台测试输入不含 `BIO_label`，32935个模板值均为空字符串|

`imcs21-cblue` 当前树以任务说明和基线/评价代码为主，没有原研究仓库的 `dataset/` 数据文件；这不等于“GitHub没有数据”。本次只对表中固定版本及本地平台包作结论，不将所有GitHub文件概括为旧版或不兼容。

## IMCS-V2的修订依据

作者原研究仓库README的News明确将“IMCS-21 2.0 version”指向 `imcs21-cblue`。后者README“更新”明确写明：

> IMCS21 数据集已更正部分标签，包括命名实体、症状标签等，并新增了 `local_implicit_info` 字段（句子级）。

同段还说明扩充4种疾病，覆盖10种儿科疾病、共4116条样本，并更新任务评价脚本及基线。本轮文件比对直接验证了标注修订和新增字段；扩充疾病是作者对版本演进的说明，不能解释为当前两个仓库之间还差4种疾病——当前原研究仓库README本身也写明4116条、10种疾病。

本轮实际打开的天池页面“更新”还给出明确日期：**2022-06-05** 新增IMCS-V2，扩充疾病类型和标注语料；**2022-12-26** 更新IMCS-V2，修复NER和SR任务标注错误，并将SR改为“局部句子级＋全局对话级”双指标。FAQ建议使用最新版，因为新版本会对历史版本做质量修正，同时另列历史版本下载入口。这些是采用平台修订版的直接官方依据。

## 本项目采用天池正式ZIP的原因

1. **统一标注版本。** 相同文本的BIO已发生实际修订，平台更新记录也明确修复NER标注；混用不同来源的金标会改变实体边界/类别及评分。全组固定平台ZIP和共享的内部train/val名单，保证各模型使用同一数据基准。
2. **对应平台任务契约。** CBLUE官方说明明确列出IMCS-V2-NER的无标签测试输入、原ID模板、预测文件名和格式检查命令。平台包直接配齐这些文件，便于本地校验及平台提交来源核对。
3. **明确评估边界。** 内部val用于调参与选模，模型确定后以官方dev作统一最终评估；官方test使用平台无标签输入与模板。GitHub论文数据、任务说明和评测代码用于来源追溯和方法核查。

本地原始ZIP为6369806字节，SHA-256：`84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67`。该哈希是本项目实算的版本标识，非平台公布的官方校验值。

## 学生认证下载流程的记录支持

有两类记录相互支持：

- **平台公开页面：** 本轮于2026-10-25实际打开页面，显示“完成学生认证，免审批下载数据集”；FAQ说明申请审核通过后可下载，文件区列出 `IMCS-V2-NER.zip`（显示上传日期2023-03-14、大小6.07MB）。这是当日公开获取流程，不代表匿名下载。
- **项目历史记录：** 2026-10-15用户明确回复“已经下载到本地”，同日已生成ZIP目录、哈希和结构审计；2026-10-16登录后的核查记录写明审核通过、学生认证有效。现存原包与当时审计哈希一致。

这些材料支持“经天池取得且当时具备学生认证下载条件”。历史材料未完整保存认证申请、审核及下载点击的逐步画面，故不补写具体认证操作步骤或凭证；公开交付仅保留上述过程摘要及数据哈希。

## 下一轮附录D.1建议文字

本项目采用IMCS-21作者接入CBLUE评测体系的IMCS-V2-NER（Task1），通过天池正式入口取得 `IMCS-V2-NER.zip` 并固定其校验值。作者原研究仓库公开了 `dataset/`，而平台2022年12月26日更新记录明确修复NER及SR标注，2.0任务仓库亦说明更正标签并增加句级字段。本项目对相同ID的train/dev进行只读比对，句子文本一致，但分别有17653句和5845句的BIO标注不同。因此，全组统一使用平台包及固定的内部train/val名单；内部val用于调参与选模，官方dev用于模型确定后的独立评估，官方test按照平台模板生成提交结果。GitHub仓库用于追溯论文、任务说明与评测代码。

## 官方来源与固定版本

- [天池CBLUE数据集页面及IMCS-V2-NER说明](https://tianchi.aliyun.com/dataset/95414)。
- [作者原研究仓库及2.0说明](https://github.com/lemuria-wchen/imcs21/blob/836048f491704011dba229a5bd04db5f7f4d2bbe/README.md#news)，本轮固定commit `836048f491704011dba229a5bd04db5f7f4d2bbe`。
- [原研究仓库dataset目录](https://github.com/lemuria-wchen/imcs21/tree/836048f491704011dba229a5bd04db5f7f4d2bbe/dataset)。
- [IMCS21-CBLUE更新说明](https://github.com/lemuria-wchen/imcs21-cblue/blob/8d512c6635029bf068712d62cf2428e65c5e4a58/README.md#更新)，本轮固定commit `8d512c6635029bf068712d62cf2428e65c5e4a58`。
- [IMCS21-CBLUE Task1说明和评价代码入口](https://github.com/lemuria-wchen/imcs21-cblue/tree/8d512c6635029bf068712d62cf2428e65c5e4a58/task1)。
- [CBLUE官方README：数据下载、IMCS-V2-NER检查与提交步骤](https://github.com/CBLUEbenchmark/CBLUE/blob/6a2c54f6a69265a33181c752a5350277149b8883/README_ZH.md)，本轮固定commit `6a2c54f6a69265a33181c752a5350277149b8883`。
- [CBLUE官方IMCS-V2-NER格式检查器](https://github.com/CBLUEbenchmark/CBLUE/blob/6a2c54f6a69265a33181c752a5350277149b8883/format_checker/format_checker_IMCS_V2_NER.py)。

比对方法：按 `(dialogue_id, sentence_id)` 对齐train/dev，比较原始文本、逐字符BIO序列及同名字段，只输出聚合计数和哈希；未改动数据、历史实验结果或正式实验报告。
