# 前期核验资料：首次发布选取

本目录选取 2026-10-15 已有的历史报告、摘要和必要格式检查参考。历史报告正文保留原始字节，平台和榜单陈述仅对应原文注明日期。当前操作以更新的交接指南及提交前实际核查为准。

- [平台规则与评估核验说明](IMCS-V2-NER_平台规则与评估核验说明.md)
- [课程全流程可行性核验](Task1_课程全流程可行性核验.md)
- [原包结构摘要](archive-audit.json)、[相关包与重叠摘要](related-packages-audit.json)、[计分契约人工探针](evaluator-contract-probes.json)
- [当时原始来源清单](source-manifest.json)：历史完整清单，列出的部分文件不在首次发布范围。
- [CBLUE 格式检查器](format_checker_IMCS_V2_NER.py)、[基类](format_checker.py)、[上游使用说明](format_checker_README.md)，固定版本与原始哈希见来源清单；[Apache-2.0 许可](CBLUE_LICENSE)仅适用于相应 CBLUE 参考文件，不代表本仓库整体或数据许可。

不分发旧榜单原始 JSON、截图、重复网页、额外任务数据或无需执行的审计脚本。历史报告中提及但未发布的材料仍在原工作目录留存；选取和字节对应关系见[发布清单](../publication_manifest.json)。这些历史材料是资料核验，不是本组真实实验。

## 测试用作者实体评分参考

[固定提交的作者评分源码](https://github.com/lemuria-wchen/imcs21-cblue/blob/8d512c6635029bf068712d62cf2428e65c5e4a58/task1/eval_task1.py)的 SHA-256 为 `9af0a1bdbdc813ff8f60b149c41e32d0d0b96f0cecb32b2bf0e37d568563d919`。上游未声明许可证，因此该源码和其 README 不复制进公开仓库，也不套用 CBLUE 的许可。

在 `evaluation/` 下运行 `python -X utf8 tests/prepare_author_reference.py`，或使用 `--source <已有副本路径>` 离线准备。校验通过的本地副本保存在本目录并被 Git 忽略。测试仅通过 AST 执行其原始实体评分段，不执行 seqeval、完整作者 CLI 或天池服务器。运行评估工具本身不依赖该参考源码。
