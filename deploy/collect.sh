#!/usr/bin/env bash
# 打包实验结果，供 scp 拉回本地。
# 用法：
#   bash collect.sh          # 只打日志和指标（几 KB）
#   bash collect.sh yes      # 连最优权重一起打（约 400MB，慢）
set -euo pipefail
cd "$(dirname "$0")"
shopt -s nullglob

TS="$(date +%Y%m%d-%H%M%S)"
OUT="results_${TS}.tar.gz"
WITH_CKPT="${1:-no}"

FILES=(runs/*/train.log runs/*/metrics.jsonl runs/*/config.json)
if [ -d code/artifacts ]; then
  FILES+=(code/artifacts)
  FILES+=(doc/实验记录.md)
fi
if [ "$WITH_CKPT" = "yes" ]; then
  FILES+=(runs/*/checkpoints/best.pt)
fi

if [ ${#FILES[@]} -eq 0 ]; then
  echo "没有找到任何实验结果，先跑 run_train.sh"
  exit 1
fi

tar czf "$OUT" "${FILES[@]}"
echo "打包完成：$OUT"
ls -lh "$OUT"
echo
echo "在你的电脑上执行（把 user@host 换成你的）："
echo "  scp user@host:~/nlp_train_pkg/$OUT ."
