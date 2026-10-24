# 复现与测试范围

从仓库的 `evaluation/` 目录按 README 的四条命令运行即可验收公开部分。Python >=3.10，标准库即可；无需 GPU、模型、网络或真实医疗数据。人工样本中的 dev 仅表示输入协议形状，不是官方 dev 的子集。所有本轮测试的原始输出与环境记录另存本地审核区，不在发布包堆入终端快照。

## 公开测试

三个原测试模块保留135项：输入校验42项、指标及评分44项、错误导出/元数据/批量流水线49项。覆盖精确边界、类型、跨句/跨对话身份、零分母、全 O、固定 macro 分母、Unicode、重复 key、非法 BIO、错误输入拒绝评分、只读输入、重复输出目录保护和 CSV 公式注入防护。

原套件有138项，其中3项执行本地固定第三方作者评分脚本。由于该源码再分发许可尚未确认，公开副本只移除其测试类及2条导入，其余断言与期望值完全保留；第三方辅助脚本不打包。本地历史138项记录与当前135项范围分别陈述。未将缺数据的测试伪装通过或用跳过掩盖依赖。

## 真实数据的受控重评分

自行合法获取固定平台版 ZIP，SHA-256 应为 `84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67`，实际许可和获取条件以当前原站为准。另向项目团队受控取得对应方法的原 dev 预测。文件放在本目录以外，例如 `../private_data/` 与 `../private_predictions/`，输出也应放在受控目录。

```bash
python -X utf8 src/evaluate_predictions.py --data ../private_data/IMCS-V2-NER.zip --predictions ../private_predictions/predictions_dev.json --split dev --purpose model_evaluation --output-dir ../private_results/rescoring
```

预期数据规模见 `results/dataset_summary.csv`，计数和未舍入分数见 `results/evaluation_summary.csv`。比较多方法时需提供真实 run_info 和 manifest，具体字段可对照合成例子与 `src/check_run_info.py`；人工例子的 null 不可直接沿用为真实运行元数据。自动核查通过只证明文件存在、哈希与字段一致，训练真实性及计时仍需人工证据。

内部划分来自官方 train：对话 ID 排序后用 `random.Random(42).shuffle`，`round(2472*0.1)=247` 个对话作 val，剩余2225个作 train。划分原代码属于共享模块，ID名单属于数据派生材料，均未收入个人候选包。这里说明算法，不伪称公开包已复现完整数据准备或包含全部训练依赖。

数据有自然文本重复：官方 dev 中9978句也见于官方 train；ID互斥不等于原文完全独立。主表保留完整官方 dev。历史选模是否曾参考 dev/平台反馈仍需团队补时间线。

2026-10-24 的材料整理只做复制、聚合、脱敏和 CPU 合成验收，不训练或新调参；真实参数表是历史训练记录。Softmax 学习率组缺各 run 原 val 预测与权重，能复核日志与均值，尚不能逐预测独立重算。规则 mc1/mc10 既有预测已在此前审计复算，其他阈值仅保存计数。天池截图到具体上传包/权重的映射尚未闭合。
