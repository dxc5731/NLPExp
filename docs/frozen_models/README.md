# 双模型冻结包：下载、身份与复现

本目录只归档说明、manifest、model_info 和**已有验收结果**，不是完整运行包。下载[完整网盘包（uknn）](https://pan.baidu.com/s/1-KdQfJJLQTruKY_d5DM3pw?pwd=uknn)后再运行包根目录的 `verify.py` / `infer.py`；这两个入口及冻结源码、Tokenizer、权重都在网盘包中，本仓库不另复制一份。

完整 ZIP：`BERT_Final_Frozen_Weights_20261027.zip`，1,398,482,444 字节，SHA-256：
`cd54d1b070d310145e40cd3f6ff74a671e5b2a8a89c91d20aa7f0b893b55ec0c`。

| 资源 | 包内路径 | 字节 | SHA-256 |
|---|---|---:|---|
| Softmax 正式微调参数 | `BERT-Softmax/weights/best.pt` | 406810499 | `93322262dd14252203966cd56499086430a51b13d1112aa0162fcf18b87f6adb` |
| CRF 第三轮 E5 | `BERT-CRF/weights/epoch_0005.pt` | 1220509451 | `e5ee6aaa8d4a00a4626553002aa9a63253a8ee0139463fec2bb41427bf391bed` |

两份 checkpoint 已包含完整微调模型参数；完整网盘包同时带匹配的 BERT 结构配置、Tokenizer、冻结源码及训练出处。此相对路径推理入口不另加载预训练本体；原仓库从头训练与旧 Softmax 入口需要预训练 BERT，不能混淆。

## 在新机器加载冻结模型

环境要求和安装命令见 [原包使用指南](使用指南.md)。该统一入口要求 CUDA；历史验收为 RTX 5060 Laptop + Python 3.13.5 / torch 2.14.1+cu132。CPU 可运行统一评分，但这里不声称冻结包推理入口支持 CPU。

在**下载后解压的完整模型包目录**，使用 ML 环境 Python：

```powershell
python -X utf8 verify.py
python -B -X utf8 infer.py --model softmax --input D:/datasets/IMCS-V2_dev.json --output outputs/session01/softmax_predictions_dev.json
python -B -X utf8 infer.py --model crf --input D:/datasets/IMCS-V2_dev.json --output outputs/session01/crf_predictions_dev.json
```

替换输入路径；重复运行使用新的 session 名，已有输出会被拒绝覆盖。模型不接收 gold，逐字符标签按原 ID 导出；同时生成 `<预测名>.run_info.json`，记录参数前后哈希、BIO 修复、耗时。随后在源码仓库按根 README 的统一校验/评分命令处理这些**新预测**。

规则 mc10 的冻结词典已在 `baseline/dict/`。完整演示工作区使用 `scripts/run_dev_demo.py --methods rule`，或全三路线命令；它调用 `backend.worker` 的原规则实现。新机器需先按 [演示说明](../demo/使用说明.md) 配置该入口的数据与证据路径；本轮未另造规则模型。

## 验收记录的时间与范围

`packaging_validation.json` 是 2026-10-27 03:38 的既有包验收。Softmax 全量 dev 68.7326 秒、CRF 104.7029 秒，均为外层执行耗时；833 对话、33,267 句、441,339 字符，逐句逐字符与正式结果相同，两模型参数前后哈希相同。不是本次仓库整理重新运行的结果。`delivery_result.json` 记录原 ZIP CRC 和权重哈希核对。

`manifest.json` 指向**完整网盘包成员**；不要对本目录执行该 manifest 的全包校验，因为此目录故意不含权重和运行源码。公开副本做了机器路径脱敏，原文件 SHA 与副本 SHA 见 [来源清单](../delivery/provenance.json)。旧内嵌哈希继续描述原包文件，不伪改历史记录。
