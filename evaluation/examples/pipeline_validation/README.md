# 三方法人工交接彩排

用途统一为 `synthetic_integration_test`。全部标签由人工构造，没有训练或执行规则、CRF、BERT 推理，不是模型表现，也不进入真实比较表。

同一份人工 gold：1 对话、2 句、22 Unicode 码点、五类各 1 个实体。第二句包含空格、中文标点、emoji、扩展汉字和组合字符。每方法均为原始双层 ID → BIO 字符串。

| 人工接口名 | TP | FP | FN | 实体 P | 实体 R | 实体 Micro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| 人工规则接口 | 0 | 0 | 5 | 0 | 0 | 0 |
| 人工CRF接口 | 2 | 2 | 3 | 1/2 | 2/5 | 4/9 |
| 人工BERT接口 | 4 | 1 | 1 | 4/5 | 4/5 | 4/5 |

CRF 人工接口故意包含一次类型差异、一次边界差异及一次漏检。BERT 人工接口故意漏掉一个实体并在另一句增加一个误检。名称仅模拟交接身份，不能解释为方法优劣。

`data_dev.json` 是小型人工金标准；三个子目录分别含预测、run_info 和人工准备记录。run_info 的设备、推理命令、耗时为 null，明确表示未运行模型；环境版本仅是准备样例时的 Python。`fixture_source.txt` 是样例来源标记，不是模型源码。

项目根目录一条命令：

```powershell
python -X utf8 src/run_evaluation_batch.py --manifest examples/pipeline_validation/evaluation_runs.json
```

输出到 `logs/pipeline_validation/batches/<新目录>/`，应为 SUCCESS、self_check=PASS，三份独立评分、每份四个错误材料文件、复现核查、总表和五类表。`comparison.json` 明示人工用途。三方法一共处理 6 条预测句子记录，只有 2 条不同的人工 gold 句子，不是 6 个独立数据样本。

自动验收：`python -X utf8 tests/run_pipeline_acceptance.py`。该命令复跑原有和新增测试、完整 CLI、官方 dev 只读审计，断言人工手算计数、独立评分与比较一致、错误 CSV 计数一致及输入未改。见[证据](../../logs/pipeline_validation/README.md)。

如需重建样例可运行 `python -X utf8 examples/pipeline_validation/build_fixture.py`；这会重写此目录内人工输入并更新环境／哈希，**不是接入真实预测所需步骤**。保留历史证据时不应随意改已有输入。真实接入请使用[指南](../../docs/真实模型预测接入与评估指南.md)，不把本目录改名当真实运行。
