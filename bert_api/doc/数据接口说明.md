# BERT 数据接口说明 v1（给成员 B）

> 你不用再碰原始 JSON。数据切分、分词对齐、DataLoader 都已经做好了，你直接用下面三行拿 batch。

## 0. 拿到包之后先做这三步

1. **解压到你的项目根目录**。解压后应该看到 `code/` 这个文件夹，而不是再套一层。
2. **装环境**：`cd code` 然后 `pip install -r requirements.txt`。
   如果装 torch 报找不到 `2.6.0+cu124`，改用
   `pip install torch --index-url https://download.pytorch.org/whl/cu124`。
3. **让程序找到数据**。两种都行：
   - 把数据集放到 `<项目根>/doc/数据/IMCS-V2-NER/IMCS-V2-NER/`（和我的结构一致）；
   - 或者设环境变量指向你放数据的目录：
     `set IMCS_DATA_DIR=D:\你的路径\IMCS-V2-NER`
4. 模型：`bert-base-chinese` 需要你自己下一次，设 `BERT_MODEL_DIR` 指向它，
   或者放到 `code/models/bert-base-chinese/`。

验证装好了：

```powershell
cd code
python -m scripts.demo_dataloader
```

看到「一致: True」就说明接口通了。

## 1. 名单在哪

`code/artifacts/split_v1.json`：

- `train`：2225 段对话 / 88842 句
- `val`：247 段对话 / 9687 句
- `seed = 42`，按**整段对话**切，官方 dev 未参与
- 源文件 SHA-256 也记在文件里，用于复现核对

**这份名单是全组唯一的切分依据**，你的 CRF 和我的 Softmax 必须用同一份，否则最后的对比不公平。

## 2. 三行拿到一个 batch

在 `code/` 目录下执行：

```python
from src.data_io import load_split
from src.dataset import build_loader, collect_examples, load_tokenizer

tok = load_tokenizer()                      # 从项目内本地目录加载，不联网
data = load_split("train")                  # 原始对话字典
examples = collect_examples(data, train_ids)  # train_ids 从 split_v1.json 读
loader = build_loader(examples, tok, batch_size=32, shuffle=True)

for batch in loader:
    ...   # batch 里有什么见下一节
```

## 3. 一个 batch 里有什么

| 键 | 形状 | 含义 |
|---|---|---|
| `input_ids` | [B, L] | 词片编号 |
| `attention_mask` | [B, L] | 1=真实词片，0=补位 |
| `labels` | [B, L] | 词片级标签；特殊符号和补位是 **-100** |
| `offsets` | [B, L, 2] | 每个词片对应原文的 [起, 止) 字符下标 |
| `dialogue_id` / `sentence_id` / `text` | list[str] | 原始 ID 和原文，导出预测时用 |

**CRF 那边特别注意**：`-100` 是"这个位置不算损失"。构造转移掩码和 loss mask 时用它，不要用 pad 的 id。特殊符号（[CLS]/[SEP]）和补位都在里面。

标签表在 `src/labels.py`，11 类，下标顺序就是模型输出维度的顺序，**不要另起一份**。

## 4. 从词片标签回到字符标签

预测出来是"每个词片一个标签"，交接要求是"每个原始字符一个标签"。用现成的：

```python
from src.align import tokens_to_char_labels
char_tags, uncovered = tokens_to_char_labels(token_label_ids, offsets[i], len(text))
```

它处理了三件容易错的事：一个词片覆盖多个字（如 `38` 是一个词片）、句首非法的 `I-X` 自动改成 `B-X`、没被覆盖的字符补 `O`。

## 5. 已经核验过什么

| 检查 | 结果 |
|---|---|
| 全量 98529 句字符覆盖 | 0 个字符丢失、0 处重叠 |
| 最长输入 | 201 词片（上限 256），0 句被截断 |
| 金标准往返 | 98529 句中 8 句（0.008%）因词片比字符粗而合并，属分词固有损失 |
| 单元测试 | 6 个，全过 |

## 6. 三条不许做的事

1. **不许用 dev 训练或调参。** dev 只用于最后交作业。
2. **不许改句子原文。** 不删空格、不归一化、不拼 speaker。标签数必须等于 `len(sentence)`。
3. **不许改 ID。** `"01"` 和 `"1"` 是两个不同的句子。
