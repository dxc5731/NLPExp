# NLPExp：中文医患对话医疗实体识别

自然语言处理课程项目，范围为 **IMCS-V2-NER / Task1**，比较规则词典、BERT-Softmax 和字符级 BERT-CRF。

当前为 **evaluation** 分支，集中展示预测交接、格式与 BIO 校验、严格实体级及字符级指标、统一重评分和结果复核。模型训练及词典构建由对应成员负责，归属见[来源与贡献](evaluation/docs/source_and_contributions.md)。

## 主要结果

以下为同一官方 dev 上的真实统一评估：833 对话、33,267 句，23,578 个金标实体。

|方法|实体 Precision|实体 Recall|实体 Micro-F1|
|---|---:|---:|---:|
|规则词典 mc10|75.6379%|77.5723%|76.5929%|
|BERT-Softmax|85.3195%|90.8813%|88.0127%|
|字符级 BERT-CRF|87.5743%|91.1994%|89.3501%|

[评估模块与图表](evaluation/README.md) · [聚合结果](evaluation/results/evaluation_summary.csv) · [参数比较](evaluation/results/parameter_comparison.csv) · [结果证据来源](evaluation/docs/results_provenance.md) · [实际工作记录](evaluation/docs/evaluation_work_summary.md)

## 离线验证

Python 3.10+，标准库、普通 CPU 即可。进入评估目录执行：

```bash
cd evaluation
python -X utf8 -m unittest discover -s tests -v
python -X utf8 src/run_evaluation_batch.py --manifest examples/pipeline_validation/evaluation_runs.json --output-dir outputs/synthetic_batch
```

预期 135 项测试通过，批量流程返回 `SUCCESS`、`self_check=PASS`。演示使用人工合成样本，演示分数不代表模型成绩。

真实数据从[天池](https://tianchi.aliyun.com/dataset/95414)自行获取并遵守许可。本次公开材料不包含原始医疗对话、完整预测、模型权重或完整训练日志；完整模型复现仍需训练方资产。评价口径、受控重评分命令和公开测试范围见[评价协议](evaluation/docs/evaluation_protocol.md)及[复现说明](evaluation/docs/reproducibility.md)。

## 训练数据研究与评分核验

[研究成果](evaluation/research/data_study/README.md)包含内部训练集聚合统计、4张PNG/SVG图、5份CSV、研究脚本与19项测试，以及八个人工评分场景的手算/本项目/作者实体评分对照。[数据来源核查](evaluation/research/data_study/天池与GitHub数据来源核查.md)说明平台ZIP与作者仓库的实际版本及标注差异。
