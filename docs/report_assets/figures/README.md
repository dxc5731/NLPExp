# 图表数据与生成说明

本目录含七张新增图表的 PNG 与 SVG，以及从原报告原样提取的 CPU 验收截图。原始医疗对话、模型权重和逐句预测不包含在交付包中。

- epoch_data.csv：四轮共 20 个完整 Epoch；R1 的 E1/E2 来自 71838655，E3–E5 来自 76608cc3。各列保留原 history.csv 数值；中断期间未完成的 Epoch 不拼接为曲线。
- plot_evidence.json：各轮最佳结果与已完成官方 dev 运行的聚合指标。
- architecture：依据第三轮冻结源码；timeline：依据真实 run 创建时间。
- crf_loss / crf_val_f1 / auxiliary_loss：未平滑或补点。
- best_and_ablation：四轮最佳结果及 R3/B1、R4/B2 对照。
- official_dev：已完成的 dev_demo_20261027_040355_4582ac8c；不是本次文档编辑期间的新推理。
- cpu_acceptance_original.png：原 DOCX 内嵌 image6.png，未重绘、未伪造；正文另以文字解释可读字段及证据限制。

重绘只需在独立文档工具环境安装 matplotlib，并运行 `python docs/report_assets/scripts/make_figures.py --output-dir ../nlpexp_redrawn_figures`。脚本默认 Microsoft YaHei 中文字体；其他系统请换成本机存在的中文字体。它只读取本交付目录的 CSV/JSON，不导入模型代码，不预测，不训练，不控制前后端。

正式报告由本人随后提交，本目录仅含报告已用图表。原始训练记录见 ../../bert_crf/training_records/；来源与原件/副本哈希见 ../../delivery/provenance.json。CPU截图对应外部原日志未找到，不把历史截图当作本次CPU独立验收。
