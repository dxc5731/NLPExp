# -*- coding: utf-8 -*-
"""把所有实验运行目录汇总成一张「实验记录」表。

老师的评分里「进度与工作量证明」要看迭代记录，这张表就是那份记录：
每次运行用了什么配置、什么时候跑的、结果多少、跑到第几轮最好。

用法（在 code/ 目录下）：
    python -m scripts.summarize_runs

输出写到 doc/实验记录.md。表是自动生成的，不要手改；
表下面「人工结论」那一节会原样保留，可以在那儿写判断和取舍。
"""

import json
from datetime import datetime
from pathlib import Path

from src.config import ROOT, RUNS

OUT = ROOT / "doc" / "实验记录.md"
MARKER = "<!-- 人工结论开始 -->"

HEADER = """# 实验记录

> 这张表由 `python -m scripts.summarize_runs` 自动生成，**不要手改上半部分**。
> 每次实验在 `runs/<run_id>/` 下留配置、训练日志、逐轮指标和最优权重，
> 表里的数字全部来自这些真实文件。

"""


def read_run(run_dir: Path) -> dict:
    info = {
        "run_id": run_dir.name,
        "created": datetime.fromtimestamp(run_dir.stat().st_ctime).strftime("%Y-%m-%d %H:%M"),
        "lr": None, "batch": None, "epochs": None, "seed": None,
        "best_f1": None, "best_epoch": None, "elapsed": None, "status": "进行中",
    }
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        info.update(lr=cfg.get("lr"), batch=cfg.get("batch_size"),
                    epochs=cfg.get("epochs"), seed=cfg.get("seed"))
        hist = data.get("history", [])
        if hist:
            best = max(hist, key=lambda h: h.get("entity_f1", -1))
            info.update(best_f1=best.get("entity_f1"), best_epoch=best.get("epoch"),
                        elapsed=best.get("elapsed_seconds"))
            info["status"] = "完成"
    else:
        mpath = run_dir / "metrics.jsonl"
        if mpath.exists():
            rows = [json.loads(x) for x in mpath.read_text(encoding="utf-8").splitlines() if x.strip()]
            if rows:
                best = max(rows, key=lambda h: h.get("entity_f1", -1))
                info.update(best_f1=best.get("entity_f1"), best_epoch=best.get("epoch"),
                            elapsed=best.get("elapsed_seconds"))
    return info


def fmt(v, digits=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def main() -> None:
    runs = sorted([p for p in RUNS.iterdir() if p.is_dir()]) if RUNS.exists() else []
    rows = [read_run(p) for p in runs]

    lines = [HEADER]
    lines.append("| run_id | 开始时间 | lr | batch | epochs | seed | 最优实体 F1 | 最优轮次 | 累计耗时(s) | 状态 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['run_id']} | {r['created']} | {fmt(r['lr'])} | {fmt(r['batch'])} | "
            f"{fmt(r['epochs'])} | {fmt(r['seed'])} | {fmt(r['best_f1'])} | "
            f"{fmt(r['best_epoch'])} | {fmt(r['elapsed'], 1)} | {r['status']} |"
        )
    if not rows:
        lines.append("| （还没有实验） | | | | | | | | | |")

    preserved = ""
    if OUT.exists():
        old = OUT.read_text(encoding="utf-8")
        if MARKER in old:
            preserved = old[old.index(MARKER):]
    if not preserved:
        preserved = (MARKER + "\n\n## 结论与取舍（人工写）\n\n"
                     "- 每次实验后在这里写：改了什么、为什么改、结果说明什么、下一步做什么。\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n" + preserved, encoding="utf-8")
    print(f"写入 {OUT}，共 {len(rows)} 次运行")


if __name__ == "__main__":
    main()
