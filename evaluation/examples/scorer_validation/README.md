# 评分器验证样例：完全人工构造

本目录中的文本、ID、金标准和预测均为人工构造。没有训练、推理或真实模型实验，不可并入正式实验成绩表。与第一阶段接口示例分开保存，是为了验证评分器在正确、漏检、类型错误和边界错误同时存在时的数值。

`data_dev.json` 与 `predictions_dev.json` 为 1 对话、2 句、22 个 Unicode 码点。第一句 15 个字符，每两个字符及后面的标点为一段；gold 依次含五类各一个实体。预测保留 Symptom、Operation，漏掉 Drug，缩短 Drug_Category，把 Medical_Examination 预测为 Drug。第二句 7 个码点全部为 O，包含空格、标点、emoji、非 BMP 汉字及组合重音符，不做归一化。

人工验算（全部为评分器验证数值）：

| 实体类别 | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Symptom | 1 | 0 | 0 | 1 | 1 | 1 |
| Drug | 0 | 1 | 1 | 0 | 0 | 0 |
| Drug_Category | 0 | 1 | 1 | 0 | 0 | 0 |
| Medical_Examination | 0 | 0 | 1 | 0 | 0 | 0 |
| Operation | 1 | 0 | 0 | 1 | 1 | 1 |
| 全局 micro | 2 | 2 | 3 | 1/2 | 2/5 | 4/9 |

gold 实体共 5 个、预测共 4 个。字符标签正确 17/22；含 O 的字符 micro-P/R/F1 均为 17/22，macro-F1 为 53/99。排除 O 类别（不删除原始位置）时，TP=5、FP=2、FN=5，micro-P=5/7、micro-R=1/2、micro-F1=10/17，macro-F1=1/2。O 标签本身 TP=12、FP=3、FN=0，F1=8/9；其余非 O 标签中 5 类 F1=1、5 类 F1=0。

在项目根目录运行：

```powershell
python -X utf8 src/evaluate_predictions.py --data examples/scorer_validation/data_dev.json --predictions examples/scorer_validation/predictions_dev.json --split dev --purpose scorer_validation --output-dir logs/evaluation/manual_cli
```

每次在指定目录下创建新子目录，不覆盖旧记录。主指标是 `entities.micro.f1`。完整定义见 [统一评分说明](../../docs/统一评分说明.md)。
