# 完全人工构造的接口示例

本目录仅验证交接格式和校验程序。文本、ID 和 BIO 均为人工构造，不来自真实 train/dev/test，不是任何模型的输出，不可作为正式实验结果或天池提交。

- `data_dev.json`：人工原始数据，2 个对话、3 个句子。
- `predictions_dev.json`：与人工数据对应的合法预测接口示例。
- `run_info.json`：人工接口元数据示例；未发生推理，因此设备、耗时等为 `null`，并在 `example_notice` 说明。真实交接不能用本例的空值代替必填运行信息。

其中 `A 😀，𠀀é` 按 Python `len(str)` 是 7 个 Unicode 码点：`A`、空格、`😀`、`，`、`𠀀`、`e`、组合重音符 U+0301。空格需要标签；组合字符不归一化。

在项目根目录运行：

```powershell
python -X utf8 src/validate_predictions.py --data examples/interface_example/data_dev.json --predictions examples/interface_example/predictions_dev.json --split dev --report logs/validation/interface_example_20261016.json
```

预期是结构校验 `PASS`，不是模型效果评价。完整约定见 [预测结果交接规范](../../docs/预测结果交接规范.md)。
