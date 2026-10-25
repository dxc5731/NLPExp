# -*- coding: utf-8 -*-
"""把内部预测转成天池提交格式。

天池要求（见《预测结果交接规范》第 5 节）：
  - 文件名必须是 IMCS-V2-NER_test.json
  - 放在提交 ZIP 的根目录，不套中间文件夹
  - 只放这一个文件；run_info、日志、权重都不进这个包

用法（在 code/ 目录下）：
    python -X utf8 -m scripts.make_tianchi_zip --predictions <predictions_test.json> --out-dir <输出目录>
"""

import argparse
import json
import zipfile
from pathlib import Path

from src.data_io import load_split
from src.labels import LABELS

SUBMIT_NAME = "IMCS-V2-NER_test.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    pred = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    data = load_split("test")

    # 与官方空模板同一个结构：对话ID -> 句子ID -> BIO 字符串
    assert set(pred) == set(data), "对话 ID 集合与 test 不一致"
    n = 0
    for did, d in data.items():
        assert set(pred[did]) == {s["sentence_id"] for s in d["dialogue"]}, f"对话 {did} 句子 ID 不一致"
        for s in d["dialogue"]:
            tags = pred[did][s["sentence_id"]].split(" ")
            assert len(tags) == len(s["sentence"]), f"{did}/{s['sentence_id']} 长度不一致"
            assert all(t in LABELS for t in tags), f"{did}/{s['sentence_id']} 非法标签"
            n += 1
    print(f"结构校验通过：{len(pred)} 段对话 / {n} 句")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / SUBMIT_NAME
    json_path.write_text(json.dumps(pred, ensure_ascii=False), encoding="utf-8")

    zip_path = out / "IMCS-V2-NER_test_submit.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, SUBMIT_NAME)

    print(f"提交文件 {json_path}  ({json_path.stat().st_size / 1e6:.1f} MB)")
    print(f"提交压缩包 {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print("包内只有这一个文件，直接上传天池即可。")


if __name__ == "__main__":
    main()
