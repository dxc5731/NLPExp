# BERT-Softmax 冻结模型本地部署与官方dev验收

完成时间：2026-10-27T02:48:34.930634+08:00。验收状态：**PASS**。

## 模型及资源

- 原训练run：`lr5e-5_ep3_20261022-151229`；真实checkpoint为Epoch3。
- Checkpoint路径：`${PROJECT}\runs\lr5e-5_ep3_20261022-151229\checkpoints\best.pt`。
- Checkpoint大小：406810499字节；SHA-256：`93322262dd14252203966cd56499086430a51b13d1112aa0162fcf18b87f6adb`，与原正式run_info完整哈希一致。
- 压缩包SHA-256：`b0f60eb207eea1d8453a713afeb481fcbc61c704d6290e4ea0095844f365e013`，给定前缀匹配；组员未给出完整预期包哈希，因此不声称核对了完整预期包哈希。
- Tokenizer/config来自本次组员包，逐文件hash见`installation.json`；预训练本体使用本机已有BERT-CRF同源资源，用户已确认一致。为原推理入口组装独立目录，无网络下载。
- `BERT_MODEL_DIR=${PROJECT}\.local\bert_softmax\NLPExp_softmax_acceptance_20261027T024216+0800\bert-base-chinese`。
- `IMCS_DATA_DIR=${PROJECT}\.local\bert_softmax\NLPExp_softmax_acceptance_20261027T024216+0800\data`；该目录为原train/dev/test逐字节核验后的受控副本。
- 原config/train.log与包内版本仅CRLF/LF差异，保留原文件；包内原字节另存证据目录`received_run`。仅补入缺失best.pt，未覆盖任何历史预测或训练文件。

## 实际新推理和验收

独立执行ID：`softmax_dev_acceptance_20261027T024216+0800`。

实际cwd：`${PROJECT}\code`。命令：

```text
"${PROJECT}\.venv-bert-crf\Scripts\python.exe" -B -X utf8 -m src.predict --run-id lr5e-5_ep3_20261022-151229 --split dev --batch-size 64 --output-dir "${PROJECT}\runs\lr5e-5_ep3_20261022-151229\softmax_dev_acceptance_20261027T024216+0800\prediction"
```

环境变量见上文及runtime_config.json。调用原`code/src/predict.py`，未修改模型算法、预测还原或训练逻辑。使用完整dev的原始句子；`require_labels=False`、`with_labels=False`，Gold不作为模型输入；独立评估阶段才使用Gold。

覆盖：833对话、33,267句、441,339个字符。完整推理、统一校验和统一评分退出码均为0。接口测试13项通过。

| 指标 | 本轮真实新预测 | 组员四位小数要求 |
|---|---:|---:|
| Precision | 0.8531953016125822 | 0.8532 |
| Recall | 0.9088133005343965 | 0.9088 |
| Micro F1 | 0.8801265068901074 | 0.8801 |
| TP / FP / FN | 21428 / 3687 / 2150 | 与原记录一致 |

新旧预测按原对话ID、句子ID和每个字符位置比较：**差异句0，差异标签0，文件字节完全相同**。

- 新预测大小：1896728字节。
- 新预测SHA-256：`f89864650eb08e1333f78781776c9c74e20ec65e815a4348ba396489eee3e51a`。
- 预测：`prediction/predictions_dev.json`。
- 运行记录：`prediction/run_info.json`与外部证据中的`commands/full_dev_inference.json`。
- 校验：`validation_dev.json`。
- 评分：`score/20261027T024656349612_bzf8acl7/evaluation.json`。
- 逐标签比较：`prediction_comparison.json`。
- 完整验收：`acceptance_result.json`。

原入口计时：含加载、分词、推理与写盘56.75秒，纯推理50.91秒；整个Python子进程62.698秒。计时范围不同，不混用。

原Softmax推理算法记录526处BIO修复，与正式旧预测相同。这是原解码流程行为；校验器不修复标签。本次没有人工修改预测。

预训练模型初始化日志中的classifier MISSING和预训练任务头UNEXPECTED来自模型初始化阶段；之后原入口通过默认strict=True的load_state_dict完整加载本次正式best.pt，成功后才推理。未执行训练；最终全量预测与原文件完全一致。

## 本机环境

Python3.13.5；PyTorch2.14.1+cu132；CUDA13.2；Transformers5.20.1；Tokenizers0.23.3；GPU为RTX5060 Laptop。环境与组员原A800/torch2.6.0+cu124不同，但本次完整输出仍逐标签、逐字节一致。

## 下次在本机重跑

运行本目录`run_new_dev.cmd`。脚本从`runtime_config.json`读取本地资源与哈希，先校验资源，再创建带时间与随机后缀的新输出目录，依次执行**推理→校验→评分**。每步保留真实日志、命令、时间和退出码，不训练、不覆盖本次或历史输出、不修改系统环境变量。

重跑脚本只做静态语法/配置检查，本轮实际验收调用的是上文原入口；没有为了测试脚本再次运行一遍完整推理。

## 保留的局限和过程记录

- 当前推理源码汇总hash与原run_info的源码汇总hash不同；实际逐文件hash及Git dirty状态已记录，不将原训练源码版本冒充本次推理版本。全量预测一致是本轮独立实证。
- 旧严格批处理的run_info schema/path适配未在本任务中修改。本次验收证明本机正式模型全量推理、格式和技术评分正常，不等于旧严格元数据批处理自动全部通过。
- 首次checkpoint只权重安全读取被旧numpy scalar元数据拒绝，之后按原入口相同的显式weights_only=False读取已核验身份的可信权重；首次接口测试因dev/test目录缺train失败，改用完整数据副本目录后13项通过。失败与修正均保留，未修改原测试或模型代码。

完整证据目录：`${LOCAL_PARENT}/NLPExp_softmax_acceptance_20261027T024216+0800`。未提交或推送Git，未启动前端开发。
