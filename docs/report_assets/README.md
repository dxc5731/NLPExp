# 报告使用的真实图表

figures 下 8 张 PNG 的字节 SHA 与当前已保存 DOCX 的 8 个内嵌图片一致；正式 DOCX/PDF 本轮不上传。7 张图同时保留 SVG。epoch_data.csv 的 20 个完整 Epoch 与四轮原 history.csv 对应；R1 分中断与恢复两个 run。

绘图脚本只读取 CSV/JSON，不导入模型、不训练或预测：

```text
python -m pip install matplotlib
python docs/report_assets/scripts/make_figures.py --output-dir ../nlpexp_redrawn_figures
```

输出目录必须不存在，防止覆盖已用于报告的原图；默认 Microsoft YaHei，可用 `--font` 指定已有中文字体。重绘后的字体/渲染器可能改变 PNG 字节，数据仍来自同一原始表。

CPU 验收图是从原报告提取的历史截图，不是本次新执行。对应外部 CPU 原日志在本次材料审计中未找到，截图可读字段之外不扩大结论。其他图的数据来源见 figures/README.md 及 ../delivery/provenance.json。
