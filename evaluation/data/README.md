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

平台发布页标注数据许可 CC BY-NC-SA 4.0；第三方代码许可与数据许可分别核对。数据来源、哈希与历史核验见[资料来源](../docs/资料来源.json)，正式使用时确认来源及许可。不要混入旧 GitHub 版本的标注或隐藏 test 的旧答案。

本目录除说明及占位文件外被 Git 忽略，原包不进入仓库或课程最终 ZIP。正式数据用途、清洗与划分由团队确定并另行记录，不覆盖原始文件；完整 dev 只读审计通过不代表模型训练、正式成绩或无泄漏验证已完成。
