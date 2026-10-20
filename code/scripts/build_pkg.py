# -*- coding: utf-8 -*-
"""打包「服务器训练包」：代码 + 数据 + 运行脚本，解压即用。

用法（在 code/ 目录下）：
    python -X utf8 -m scripts.build_pkg

产出 handoff/nlp_train_pkg.tar.gz。服务器上解压后按 README_服务器操作.md 操作。
"""

import shutil
import tarfile
from pathlib import Path

from src.config import CODE, DATA_DIR, ROOT

PKG_NAME = "nlp_train_pkg"
STAGE = ROOT / "handoff" / "_pkg"
OUT = ROOT / "handoff" / f"{PKG_NAME}.tar.gz"

CODE_ITEMS = ["src", "scripts", "tests", "artifacts"]
DATA_FILES = ["IMCS-V2_train.json", "IMCS-V2_dev.json", "mappings.json", "README.txt"]


def copy_tree(src: Path, dst: Path, skip=("__pycache__",)) -> None:
    for p in src.rglob("*"):
        if any(s in p.parts for s in skip):
            continue
        rel = p.relative_to(src)
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    pkg = STAGE / PKG_NAME
    pkg.mkdir(parents=True)

    # 1. 代码
    for item in CODE_ITEMS:
        src = CODE / item
        if not src.exists():
            continue
        copy_tree(src, pkg / "code" / item)
    shutil.copy2(CODE / "requirements-server.txt", pkg / "code" / "requirements-server.txt")
    (pkg / "code" / "scripts" / "build_pkg.py").unlink(missing_ok=True)

    # 2. 数据（保持和本地一致的目录层级，config.py 才找得到）
    data_dst = pkg / "doc" / "数据" / "IMCS-V2-NER" / "IMCS-V2-NER"
    data_dst.mkdir(parents=True)
    for name in DATA_FILES:
        shutil.copy2(DATA_DIR / name, data_dst / name)

    # 3. 脚本和说明
    for p in (ROOT / "deploy").iterdir():
        if p.is_file():
            shutil.copy2(p, pkg / p.name)

    # 4. 打包
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(str(pkg), arcname=PKG_NAME)
    shutil.rmtree(STAGE)

    size_mb = OUT.stat().st_size / 1e6
    print(f"打包完成：{OUT}  ({size_mb:.1f} MB)")
    print("服务器上解压后会看到：")
    for p in sorted((Path(OUT).parent).glob("*")):
        pass
    print("  nlp_train_pkg/  ├─ setup.sh  ├─ run_train.sh  ├─ collect.sh  ├─ code/  └─ doc/数据/")


if __name__ == "__main__":
    main()
