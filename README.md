# NLPExp：中文医患对话医疗实体识别

自然语言处理课程项目，方向 B：信息抽取／序列标注，当前范围为 **IMCS-V2-NER / Task1**。

## 评估模块入口

当前分支为 **evaluation**，存放评估、比较、复现核查与交接工具。评估内容集中在 [evaluation/](evaluation/README.md)，模型训练与预测由对应成员维护。稳定的共享内容后续通过 PR 合并到 main。

| 入口 | 内容 |
|---|---|
| [评估说明与命令](evaluation/README.md) | 环境、准备步骤、人工验证、真实预测接入 |
| [预测结果交接规范](evaluation/docs/预测结果交接规范.md) | 队友交付文件、字符对齐和元数据约定 |
| [统一评分说明](evaluation/docs/统一评分说明.md) | 严格实体 P/R/F1、逐字符指标和验证边界 |
| [真实预测接入指南](evaluation/docs/真实模型预测接入与评估指南.md) | 批量清单、一条命令、结果位置和失败定位 |
| [进度与首次入库说明](evaluation/docs/仓库发布说明.md) | 已完成、待完成、历史日志与提交记录的关系 |
| [课程原始材料](docs/course/README.md) | 任务书、报告模板、个人贡献声明模板 |
| [AI 交互记录](evaluation/AI交互记录.md) | 用户决定、AI 辅助及实际验证记录 |

评估工具已在本地完成前三阶段工程开发和人工样例验证；**尚无本组真实模型预测或正式模型成绩**。历史工作日期保存在原始日志中，Git 记录从实际首次入库日期开始。草案及实验设置的待确认状态保留，人工验证数值不作为模型表现。

## 快速验证

Python 3.10+，评估运行只依赖标准库、普通 CPU。以下步骤不需要数据集、模型权重、GPU 或 API Key；首次准备测试参考时需要网络。

```powershell
git clone --branch evaluation https://github.com/dxc5731/NLPExp.git
cd NLPExp/evaluation
python -X utf8 tests/prepare_author_reference.py
python -X utf8 -m unittest discover -s tests -v
python -X utf8 src/run_evaluation_batch.py --manifest examples/pipeline_validation/evaluation_runs.json --csv-bom
```

第一条 Python 命令下载固定版本的公开作者评分脚本并核对 SHA-256，供测试提取其实体评分代码段；该第三方脚本不在本仓库分发，不是本组实现。预期测试通过，人工批量运行返回 `SUCCESS`、`self_check=PASS`，输出在 `evaluation/logs/pipeline_validation/batches/`。这些是工程验证，真实模型接入还须检查实际文件、配置、日志及人工复核事项。

真实数据准备和完整验收命令见[评估模块说明](evaluation/README.md)。原始数据、预测、权重、环境和医疗原文分析保留在本地；公开仓库保留工具、模板及经过选择的验证证据。
