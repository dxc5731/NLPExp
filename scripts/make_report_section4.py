# -*- coding: utf-8 -*-
"""生成报告 ④ 实现细节 的 docx 片段，供佳佳直接粘贴进主报告。"""
from docx import Document
from docx.shared import Pt

doc = Document()

# 与主报告一致的样式：Heading 1 / Heading 2 / Normal
doc.add_heading("④ 实现细节", level=1)
doc.add_paragraph(
    "本节说明公共数据处理与 BERT-Softmax 方案的实现，包含数据管线、模型与训练配置、"
    "实现中真实遇到的问题及其处理方式，以及运行环境与工程约束。"
)

doc.add_heading("4.1 公共数据管线", level=2)
doc.add_paragraph(
    "数据划分以官方 train 为来源，按整段对话切分而非按句子切分。同一段对话中的句子共享"
    "病人、医生与表达习惯，相似度远高于随机两句；若按句子随机切分，近似句子会同时落入训练"
    "与验证两侧，验证分数虚高且不会报错。划分时先对对话 ID 排序，再以固定种子 42 打乱，"
    "取 10% 作为内部验证集，其余为训练集，名单固化为 split_v1.json 供全组三种方法共用。"
)
doc.add_paragraph(
    "分词与字符级 BIO 的对应关系通过 tokenizer 返回的字符区间（offset）建立。中文多数情况下"
    "一个字符对应一个词片，但数字、英文与符号会被合并：实测“38”是一个词片，覆盖“3”和“8”"
    "两个字符。若按词片首字符取标签，跨实体的词片会被截断。最终规则是：只有当词片标签能够"
    "逐字还原其覆盖的所有字符时才使用该标签，否则将该位置标为 -100 不参与损失计算，并记入"
    "歧义清单。全量 98,529 句检查结果为：0 个字符未被覆盖、0 处区间重叠、0 句被截断；"
    "8 个词片无法无损映射（涉及 8 句、21 个字符），已逐条列出，其余 98,521 句可逐字还原。"
)
doc.add_paragraph(
    "特殊符号与补齐位不参与训练与预测；导出前只对预测结果修复非法 BIO 转移，并记录修复处数。"
    "该数据集最长句为 199 个字符（201 个词片），远小于 256 的上限，因此不实现滑窗；"
    "编码时关闭截断，超长输入直接抛出异常，避免静默截断后半句。"
)

doc.add_heading("4.2 模型与训练配置", level=2)
doc.add_paragraph(
    "模型为 bert-base-chinese（词表 21,128，约 1.017 亿参数）接一层 11 维线性分类头，"
    "逐位置独立分类后取 argmax；与 BERT-CRF 的差别在于没有显式标签转移约束，"
    "因此可能出现句首 I-X 一类的非法序列，需在解码阶段修复。"
)
table = doc.add_table(rows=1, cols=2)
table.style = "Table Grid"
hdr = table.rows[0].cells
hdr[0].text = "配置项"
hdr[1].text = "取值"
rows = [
    ("预训练模型", "bert-base-chinese"),
    ("最大长度", "256（实际最长 201 词片）"),
    ("batch size", "32"),
    ("训练轮数", "3"),
    ("学习率", "5e-5（另比较 2e-5、3e-5）"),
    ("学习率调度", "线性预热 10% + 线性衰减"),
    ("优化器", "AdamW，weight decay 0.01，梯度裁剪 1.0"),
    ("dropout", "0.1"),
    ("随机种子", "42（重复实验另用 1、2）"),
    ("确定性", "固定 Python/NumPy/PyTorch 种子，开启 cudnn.deterministic"),
]
for k, v in rows:
    cells = table.add_row().cells
    cells[0].text = k
    cells[1].text = v

doc.add_paragraph(
    "模型选择只使用内部验证集，官方 dev 全程未参与训练或调参。每次运行保存配置、训练日志、"
    "逐轮指标与最优 checkpoint；预测阶段输出 predictions_dev.json 与 run_info.json，"
    "后者记录数据与预测文件的哈希、代码快照哈希、环境版本与实测推理耗时。"
)

doc.add_heading("4.3 实现中遇到的问题与处理", level=2)
doc.add_paragraph(
    "以下五项均在实际运行中出现过，处理方式与对应的证据可在运行日志和提交记录中核对。"
)
t = doc.add_table(rows=1, cols=3)
t.style = "Table Grid"
h = t.rows[0].cells
h[0].text = "问题"
h[1].text = "现象"
h[2].text = "处理方式"
items = [
    ("一个词片覆盖多个字符",
     "“38”是一个词片对应两个字符，按首字符取标签会把跨实体边界处的标签截断",
     "改为“能逐字还原才使用，否则标 -100 并记入歧义清单”，全量 98,529 句复检"),
    ("镜像站不支持 Xet 协议",
     "模型下载返回 404，随后进程段错误退出",
     "关闭 Xet 传输，改用镜像的直链逐文件下载，并在文档中记录该限制"),
    ("pip 缓存目录不可写",
     "安装依赖时报 PermissionError，指向系统受管目录",
     "关闭缓存安装，同时关闭 pip 版本自检，避免联网阻塞"),
    ("显存容量受限",
     "8GB 显存的笔记本显卡同时运行桌面应用时可用显存更少，训练统一使用 batch 32",
     "固定 batch 32 并记录该约束；同配置换到 云服务器后三轮训练由 71 分钟降至 11 分钟"),
    ("多进程并行训练的反常耗时",
     "四组并行耗时 48 分钟，而三组并行仅 11 分钟",
     "定位瓶颈在 CPU 分词而非 GPU，改为按三组并行排期"),
]
for a, b, c in items:
    cells = t.add_row().cells
    cells[0].text = a
    cells[1].text = b
    cells[2].text = c

doc.add_paragraph(
    "并行训练的反常耗时一项说明：每个训练进程都独立读取并分词 8.8 万句，四进程时 CPU 成为"
    "瓶颈，GPU 利用率反而下降。改为三组并行后，单组耗时回到 11 分钟。"
)

doc.add_heading("4.4 运行环境与工程约束", level=2)
doc.add_paragraph(
    "训练在本地环境与云服务器上完成，两处均使用 Python 3.11 / 3.13.5、"
    "PyTorch 2.6.0+cu124 与 transformers 5.20.1。同一配置在两处环境上的验证集实体 "
    "F1 相差 0.0006，说明种子与确定性设置有效。"
)
doc.add_paragraph(
    "数据与权重均不进入仓库：仓库只保存源码、配置、切分名单、训练日志与预测结果，"
    "权重通过路径与 SHA-256 标识。推理与训练均本地离线完成，不调用任何外部 API，"
    "因此不存在 token 费用与患者文本外传问题。"
)

doc.save(r"D:\agent-chat\dsh-deepseek\nlp\handoff\报告片段_④实现细节_ltm.docx")
print("已生成")
