# 先看这个

「BERT 数据接口 v1」交接包，给成员 B。

## 里面有什么
- `doc/数据接口说明.md`  ← 先读这个
- `code/artifacts/split_v1.json`  ← 全组共用的 train/val 名单（seed=42, 90:10）
- `code/src/`  ← 分词对齐、Dataset、标签表
- `code/tests/`  ← 自动测试
- `code/scripts/demo_dataloader.py`  ← 一条命令验证接口通没通
- `code/requirements.txt`  ← 环境

## 三步
1. 解压到项目根目录（解压后应该看到 `code/`，不要再套一层）
2. `cd code` 然后 `pip install -r requirements.txt`
3. `python -m scripts.demo_dataloader` → 看到 `一致: True` 就成了

数据放哪、模型放哪，见说明文档第 0 节。
