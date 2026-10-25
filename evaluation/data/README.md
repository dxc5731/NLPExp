# 本地数据准备

从[天池 CBLUE 数据入口](https://tianchi.aliyun.com/dataset/95414)按实际账号权限获取平台版 **IMCS-V2-NER.zip**。下载／申请可能需要登录；本仓库不分发原始数据，不承诺匿名下载可用。

建议把包放在当前模块 `data/IMCS-V2-NER.zip`，也可在 `--official-data`、`--data` 或本地清单中填写其他实际路径。源码支持只读 ZIP，不要求解压或修改原包。

本组已核验的固定原包 SHA-256：

```text
84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67
```

```powershell
(Get-FileHash -LiteralPath 'data/IMCS-V2-NER.zip' -Algorithm SHA256).Hash.ToLowerInvariant()
```

平台发布页标注数据许可 CC BY-NC-SA 4.0；第三方代码许可与数据许可分别核对。数据来源、哈希与历史核验见[资料来源](../docs/资料来源.json)，正式使用时确认来源及许可。本项目固定采用天池修订包。按ID比对已核对的作者原仓库固定版本后，train/dev句子文本相同，但分别有17,653句和5,845句的BIO标注不同，因此复评应使用同一平台金标。GitHub用于任务说明与评分实现追溯，具体版本及聚合差异见[来源比对记录](../research/data_study/天池与GitHub数据来源核查.md)。官方test使用平台无标签输入，不混入其他来源的测试答案。

本目录除说明及占位文件外被 Git 忽略，原包不进入仓库或课程最终 ZIP。全组已从官方train按完整对话固定划分（seed=42）：internal_train为2,225组对话、88,842句，用于模型训练与词典等训练资源构建；internal_val为247组对话、9,687句，用于调参与模型选择。官方dev用于方案确定后的统一最终评估；官方test无本地BIO金标，用于平台预测提交。固定划分名单在本地保存，读取与格式校验保留原文及字符标签，不覆盖原始数据文件。
