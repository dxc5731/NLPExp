#!/usr/bin/env bash
# 跑一次训练。
# 用法：
#   bash run_train.sh                 # 默认 lr=3e-5，3 轮
#   bash run_train.sh 5e-5            # 换学习率
#   bash run_train.sh 5e-5 3 my_run   # 自定义 run_id
# 输出在 code/runs/<run_id>/，日志实时打印，同时写进 train.log。
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate

LR="${1:-3e-5}"
EPOCHS="${2:-3}"
RUN_ID="${3:-lr${LR}_ep${EPOCHS}_$(date +%Y%m%d-%H%M%S)}"

export HF_HUB_OFFLINE=1
cd code
echo "开始训练：lr=$LR  epochs=$EPOCHS  run_id=$RUN_ID"
python -X utf8 -m src.train --lr "$LR" --epochs "$EPOCHS" --run-id "$RUN_ID"
echo
echo "训练结束。结果在 code/runs/$RUN_ID/"
echo "下一步： bash collect.sh"
