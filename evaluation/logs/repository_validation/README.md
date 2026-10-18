# 仓库目录中的重新验证

2026-10-18 在独立 Git 检出的 `evaluation/` 目录实跑；Windows、Python 3.14.0、标准库 CPU。实际环境、命令、退出码、耗时和实现哈希见[verification.json](20261018/verification.json)。

| 检查 | 结果与证据 |
|---|---|
| 固定作者参考准备 | 哈希校验 PASS；[命令输出](20261018/prepare_author_reference.txt)；参考源码被 Git 忽略 |
| 分阶段及全量测试 | 42、47、49；全量 138 项通过，0 失败；[完整入口输出](20261018/acceptance_cli.txt) |
| 三方法人工完整流程 | SUCCESS，self_check=PASS；独立评分、比较表、错误计数与文件复读一致 |
| 官方 dev 只读核验 | 833 对话／33267 句／441339 Unicode 码点，错误 0；原包哈希及人工输入未改变 |
| 作者实体段交叉验证 | 四组人工场景及五类 F1 全部通过；[命令输出](20261018/author_crosscheck.txt)、[结果](20261018/author_crosscheck.json) |

完整新验收：[acceptance_summary.json](../pipeline_validation/acceptance/20261018T225551849836_46_l2go2/acceptance_summary.json)。同目录保留各阶段及全量测试输出、人工三方法 CLI、官方 dev 只读报告，以及相应人工批次的全部计分和错误／比较产物。

人工金标准仅 2 句，三方法共 6 条预测句记录。这是仓库迁入后的工程验证，没有真实训练、推理、模型成绩或完整 dev 的效果评分；也未证明跨 Python 版本可运行。命令所用数据路径是本次真实本机位置；其他机器按 README 准备各自数据路径。

历史最终验收仍单独保存在 2026-10-17 目录，不改写旧结果为本次运行。公开发布前的文件、链接、UTF-8、来源哈希和忽略规则检查见[publication_checks.json](20261018/publication_checks.json)。

## 从 GitHub 分支重新克隆后的验证

首次推送后，重新从公开 `evaluation` 分支克隆提交 `603885a77d152c8ad44f9441a28c1223e39049a6`，核对发布清单的 114 项文件哈希，确认没有依赖原检出未提交的作者参考文件。随后按根 README 重新下载并核对固定参考，实跑全量 138 项测试及三方法人工 CLI，均通过，人工批次 SUCCESS、self_check=PASS。

证据：[fresh_clone_verification.json](20261018/fresh_clone_verification.json)、[准备输出](20261018/fresh_clone_prepare.txt)、[全量测试](20261018/fresh_clone_tests.txt)、[人工 CLI](20261018/fresh_clone_synthetic_cli.txt)。这次复检仍在同一 Windows／Python 主机，只验证从仓库获取的文件和公开步骤，不构成其他操作系统、Python 版本或真实模型的验证。
