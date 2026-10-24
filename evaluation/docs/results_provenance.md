# 结果字段与证据层级

CSV 中所有 precision/recall/f1/accuracy 为0—1比例，显示百分数需乘100；差值的百分点需再乘100。不将内部val、官方dev或平台test混合排名。

|文件|集合和用途|来源与边界|
|---|---|---|
|evaluation_summary.csv / entity_type_summary.csv|完整官方dev；三方法主指标、字符辅助及15行分类指标|E024/E027/E030原评分与E136/E137/E138既有重评分逐字段一致；本轮仅读取核对与聚合，未启动模型|
|dataset_summary.csv|官方train/dev/test及seed42内部划分|固定平台包与现有真实划分审计；test金标实体数为not_public|
|parameter_comparison.csv|内部val；六个Softmax学习率run及两行三seed均值|原config与metrics直接读取，原train.log此前已核对；每seed内仅lr不同，batch32/3epoch/dropout0.1/max_len256/wd0.01/warmup0.1保持一致。均选E3。缺原val预测/权重，未重新训练|
|rule_threshold_comparison.csv|内部val；6个min_count点|E072保存计数；mc1与mc10原预测此前已复算，其余四点仅保存计数。本包不包含训练词面词典|
|softmax_epoch_history.csv|内部val；正式Softmax三轮|E054原metrics的三个真实点。loss为有效token的CE按batch均值再平均；不是字符加权loss，也不是CRF的NLL；累计秒不是单次请求延迟|
|tianchi_reported_results.csv|隐藏金标官方test；外部补充|2026-10-24用户截图U001；本轮从现有报告提取内嵌图，实看核对三行NER P/R/F1及完成状态。figures/tianchi_reported_test_results.png为去元数据副本，未改可见内容。缺平台原始导出/上传回执，不绑定具体checkpoint，不发布推断排名|
|verification_history.json|2026-10-17合成工程验收|原摘要的精选字段与原件SHA；不当作真实模型训练或性能结果|

Softmax三seed均值为88.5858%与88.8397%，样本标准差为0.0518与0.0178个百分点，均值差0.2539个百分点。正式dev Softmax与CRF的整套流程并非相同超参数和字符监督，因此1.3375个百分点差值不解释为单独CRF层的因果提升。

五类支持数在三方法中相同，合计23578。实体与字符统计来自同一dev：Softmax字符正确428340个，CRF428335个；实体TP为21428与21503。图中数值只为显示四舍五入；应以CSV未舍入值为准。

原参数运行和曲线属于模型负责人，统一评分和聚合复核属于评估工作。`provenance.json` 提供原件SHA和匿名证据ID，供持有原资料的人校验；哈希不是公开原始文件的替代品，也不能让缺资产的读者直接复现训练。未收录耗时公平排名；现有设备和计时范围不一致。商业API token计费不适用，计算资源费用未取得账单。
