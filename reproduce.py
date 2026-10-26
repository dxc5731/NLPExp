# -*- coding: utf-8 -*-
"""一键复现脚本：检查环境 -> 单元测试 -> 训练 -> dev 预测 -> 官方校验 -> 官方评分。

用法（在仓库根目录）：

    python reproduce.py --check     # 只检查环境，不训练（约 5 秒）
    python reproduce.py --quick     # 快速复现：100 段对话 1 轮（约 1—2 分钟）
    python reproduce.py --full      # 全量复现：88842 句 3 轮（云服务器约 11 分钟，本地 GPU 约 71 分钟）
    python reproduce.py --download-model   # 只下载预训练模型到 code/models/

所有命令都用当前 Python 解释器执行（`sys.executable`），
所以请先激活装了依赖的环境，或用 `./.venv/Scripts/python.exe reproduce.py`。

退出码：0 = 全部通过；1 = 有步骤失败（会指出卡在哪一步）。
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
EVALUATION = ROOT / "evaluation"

SEED = 42
RUN_ID = "reproduce"
ARCHIVE_SHA256 = "84e34d2417b839d9fbe5c5f391e44de2a9b01bd5f31d576b8feccc78c1dfbc67"

DATA_HASHES = {
    "IMCS-V2_train.json": "07cdd3799087283a5a17de8bb43e71cb159a9a2c99c1cfbe8081847fac358f24",
    "IMCS-V2_dev.json": "16f597ad368971c7db994ca904e05d91942f0e3c73009c9ef1d597fde065d39f",
    "IMCS-V2_test.json": "46ee2e8bbffdc722cad7b813e5d8223c5842904924a164faaf0db523e7c0f09d",
}

# 预期值（同环境、同种子下的实测结果；用于判断"复现成功"）
# quick 用的是 lr 5e-5、100 段对话、1 轮，数值本身不代表模型效果，只用来确认流程通
EXPECT = {
    "quick": {"f1": 0.7228, "f1_tol": 0.06, "minutes": "1—2"},
    "full": {"f1": 0.8885, "f1_tol": 0.005, "minutes": "11（云服务器）/ 71（本地 GPU）"},
}

_results = []


def say(msg=""):
    print(msg, flush=True)


def record(stage, expect, actual, ok):
    _results.append((stage, expect, actual, "PASS" if ok else "FAIL"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, cwd=None, timeout=None):
    """跑一条命令，实时打印，返回 (成功?, 输出)。"""
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True,
                          capture_output=True, timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out


def data_dir() -> Path:
    env = os.environ.get("IMCS_DATA_DIR")
    return Path(env) if env else ROOT / "doc" / "数据" / "IMCS-V2-NER" / "IMCS-V2-NER"


def model_dir() -> Path:
    env = os.environ.get("BERT_MODEL_DIR")
    return Path(env) if env else CODE / "models" / "bert-base-chinese"


def check_env(strict_model=True) -> bool:
    say("=" * 66)
    say("1. 环境检查")
    ok = True
    say(f"   Python {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 9):
        say("   ✗ 需要 Python 3.9+")
        ok = False

    for mod in ("torch", "transformers", "seqeval", "numpy"):
        try:
            m = __import__(mod)
            say(f"   {mod:14s} {getattr(m, '__version__', '?')}")
        except Exception as e:
            say(f"   ✗ {mod} 未安装：{e}")
            ok = False

    try:
        import torch
        say(f"   CUDA 可用: {torch.cuda.is_available()}"
            + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "  （CPU 也能跑，只是慢）"))
    except Exception:
        pass

    say("\n2. 数据检查")
    d = data_dir()
    say(f"   数据目录：{d}")
    if not d.exists():
        say("   ✗ 找不到数据目录。请从 https://tianchi.aliyun.com/dataset/95414 下载 IMCS-V2-NER.zip，")
        say(f"     解压到 {d}")
        say("     或设置环境变量 IMCS_DATA_DIR 指向含 JSON 的目录。")
        ok = False
    else:
        for name, expect_hash in DATA_HASHES.items():
            p = d / name
            if not p.exists():
                say(f"   {name:24s} 缺失")
                if name != "IMCS-V2_test.json":      # test 只有最终提交才需要
                    ok = False
                continue
            got = sha256(p)
            mark = "✓" if got == expect_hash else "✗ 哈希不符（可能是别的版本）"
            say(f"   {name:24s} {p.stat().st_size / 1e6:7.1f} MB  {mark}")
            if got != expect_hash and name != "IMCS-V2_test.json":
                ok = False

    say("\n3. 模型检查")
    m = model_dir()
    say(f"   模型目录：{m}")
    needed = ["config.json", "vocab.txt"]
    has_weight = any((m / n).exists() for n in ("model.safetensors", "pytorch_model.bin"))
    missing = [n for n in needed if not (m / n).exists()]
    if missing or not has_weight:
        detail = "、".join(missing) if missing else "权重文件（model.safetensors 或 pytorch_model.bin）"
        say(f"   ✗ 缺少：{detail}")
        if strict_model:
            say("     下载：python reproduce.py --download-model")
            say("     或设置环境变量 BERT_MODEL_DIR 指向已下载的模型目录。")
            ok = False
    else:
        say("   ✓ config / 词表 / 权重都在")
    return ok


def download_model() -> int:
    say("下载 bert-base-chinese 到 code/models/bert-base-chinese …")
    target = CODE / "models" / "bert-base-chinese"
    os.environ.setdefault("HF_HOME", str(CODE / ".hf-cache"))
    code = (
        "from huggingface_hub import snapshot_download;"
        f"snapshot_download('bert-base-chinese', local_dir=r'{target}',"
        " allow_patterns=['*.json','*.txt','*.bin','*.safetensors'])"
    )
    ok, out = run([sys.executable, "-c", code])
    if ok:
        say(f"完成：{target}")
        return 0
    say("直连失败，改用镜像重试……")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    ok, out = run([sys.executable, "-c", code])
    if ok:
        say(f"完成：{target}")
        return 0
    say(out[-2000:])
    return 1


def run_tests() -> bool:
    say("\n4. 单元测试")
    ok, out = run([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-q"], cwd=CODE)
    tail = [l for l in out.strip().splitlines() if l.strip()][-3:]
    for l in tail:
        say("   " + l)
    record("单元测试", "13 passed", tail[-1] if tail else "无输出", ok)
    return ok


def run_train(mode: str) -> bool:
    say(f"\n5. 训练（{mode}）")
    cmd = [sys.executable, "-X", "utf8", "-m", "src.train",
           "--lr", "5e-5", "--epochs", "3" if mode == "full" else "1",
           "--seed", str(SEED), "--run-id", RUN_ID]
    if mode == "quick":
        cmd += ["--limit-dialogues", "100"]
    say("   " + " ".join(cmd[1:]))
    ok, out = run(cmd, cwd=CODE)
    f1 = None
    for line in out.splitlines():
        if line.startswith("epoch ") and "实体P" in line:
            say("   " + line)
        if line.startswith("完成，总耗时"):
            say("   " + line)
        if "最优实体 F1" in line and "完成" in line:
            try:
                f1 = float(line.split("最优实体 F1")[1].strip())
            except Exception:
                pass
    if not ok:
        say(out[-2000:])
        record("训练", "跑通", "失败", False)
        return False
    exp = EXPECT[mode]
    same = f1 is not None and abs(f1 - exp["f1"]) <= exp["f1_tol"]
    record("训练（验证集实体 F1）", f"{exp['f1']:.4f} ± {exp['f1_tol']}",
           f"{f1:.4f}" if f1 is not None else "未解析到", same)
    # 数值不合预期不阻断后续：训练本身成功就继续，让预测/校验/评分也跑一遍
    return ok


def run_predict() -> bool:
    say("\n6. dev 预测")
    ok, out = run([sys.executable, "-X", "utf8", "-m", "src.predict",
                   "--run-id", RUN_ID, "--split", "dev"], cwd=CODE)
    for line in out.splitlines():
        if line.startswith("自检通过") or line.startswith("非法 BIO"):
            say("   " + line)
    record("dev 预测", "833 对话 / 33267 句", "见上" if ok else "失败", ok)
    return ok


def run_validate(pred: Path) -> bool:
    say("\n7. 官方校验器")
    report = ROOT / "runs" / RUN_ID / "validation_report.json"
    ok, out = run([sys.executable, "-X", "utf8", "src/validate_predictions.py",
                   "--data", str(data_dir()), "--predictions", str(pred),
                   "--split", "dev", "--report", str(report)], cwd=EVALUATION)
    line = next((l for l in out.splitlines() if l.startswith("PASS") or l.startswith("FAIL")), "")
    say("   " + (line or out[-500:]))
    record("官方校验器", "PASS", line[:4] if line else "无输出", ok)
    return ok


def run_score(pred: Path, mode: str) -> bool:
    say("\n8. 官方评分器")
    out_dir = ROOT / "runs" / RUN_ID / "score"
    ok, out = run([sys.executable, "-X", "utf8", "src/evaluate_predictions.py",
                   "--data", str(data_dir()), "--predictions", str(pred),
                   "--split", "dev", "--purpose", "model_evaluation",
                   "--output-dir", str(out_dir)], cwd=EVALUATION)
    line = next((l for l in out.splitlines() if "ENTITY MICRO" in l), "")
    say("   " + (line or out[-500:]))
    f1 = None
    if "F1=" in line:
        try:
            f1 = float(line.split("F1=")[1].split()[0])
        except Exception:
            pass
    if mode == "quick":
        # quick 的模型只在 100 段对话上训练过，分数没有意义，只确认评分器能跑
        record("官方评分器", "能跑出结果", f"F1={f1:.4f}" if f1 else "未解析到", ok)
        return ok
    ok_f1 = f1 is not None and abs(f1 - 0.8801) <= 0.005
    record("dev 实体 F1", "0.8801 ± 0.005", f"{f1:.4f}" if f1 else "未解析到", ok_f1)
    return ok and ok_f1


def summary(mode: str) -> int:
    say("\n" + "=" * 66)
    say("复现结果汇总")
    say("-" * 66)
    say(f"{'阶段':<26}{'预期':<24}{'实际':<16}结果")
    for stage, expect, actual, status in _results:
        say(f"{stage:<24}{expect:<24}{str(actual):<16}{status}")
    say("-" * 66)
    bad = [r for r in _results if r[3] == "FAIL"]
    if bad:
        say(f"有 {len(bad)} 项未通过，见上表。")
        return 1
    say(f"全部通过。（{mode} 模式）")
    say("注意：quick 只证明流程通，主结果请用 --full。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="只检查环境")
    g.add_argument("--quick", action="store_true", help="快速复现（约 1—2 分钟）")
    g.add_argument("--full", action="store_true", help="全量复现（20—70 分钟）")
    g.add_argument("--download-model", action="store_true", help="下载预训练模型")
    args = ap.parse_args()

    if args.download_model:
        return download_model()

    mode = "full" if args.full else "quick"
    t0 = time.time()
    if not check_env():
        say("\n环境检查未通过，先解决上面的问题。")
        return 1
    if args.check:
        say("\n环境检查通过。下一步：python reproduce.py --quick")
        return 0

    say(f"\n预计耗时：{EXPECT[mode]['minutes']} 分钟（{mode} 模式，seed={SEED}）")
    if mode == "full":
        say("全量训练会长时间占用 GPU，随时可以 Ctrl+C 中断。")

    run_tests()
    if not run_train(mode):
        return summary(mode)
    if not run_predict():
        return summary(mode)

    pred = ROOT / "runs" / RUN_ID / "predict_dev" / "predictions_dev.json"
    run_validate(pred)
    run_score(pred, mode)
    say(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return summary(mode)


if __name__ == "__main__":
    sys.exit(main())
