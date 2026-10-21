#!/usr/bin/env bash
# 并行跑多组实验，每组占一张卡。
# 用法（在包根目录）：
#   bash run_parallel.sh 2e-5 3e-5 5e-5
# 每组一个独立进程、独立 run_id，日志各写一份 run_<lr>.log。
# 电脑/服务器有 8 张卡，三层实验 30 分钟内能全出结果。
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate

if [ "$#" -eq 0 ]; then
  echo "用法: bash run_parallel.sh <lr1> [lr2] [lr3] ..."
  echo "例:   bash run_parallel.sh 2e-5 3e-5 5e-5"
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
cd code
pids=()
i=0
run_ids=()

for lr in "$@"; do
  run_id="lr${lr}_ep3_${TS}"
  run_ids+=("$run_id")
  echo "GPU $i -> lr=$lr  run_id=$run_id  日志 run_${lr}.log"
  CUDA_VISIBLE_DEVICES=$i HF_HUB_OFFLINE=1 \
    python -X utf8 -m src.train --lr "$lr" --epochs 3 --run-id "$run_id" \
    > "run_${lr}.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
done

echo "已启动 ${#pids[@]} 个训练进程，等待全部结束……"
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

echo
echo "全部结束。各组的最后一行："
for lr in "$@"; do
  tail -n 2 "run_${lr}.log" | head -n 1
done

if [ "$fail" -ne 0 ]; then
  echo
  echo "有进程失败，检查 run_*.log 里的报错。"
  exit 1
fi

echo
echo "下一步： bash collect.sh"
