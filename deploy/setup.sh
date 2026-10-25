#!/usr/bin/env bash
# 一次性环境准备：建虚拟环境 -> 装 torch -> 装其余依赖 -> 下模型 -> 自检
# 用法：bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

echo "== 1/5 检查 Python =="
"$PY" -V
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -V

echo "== 2/5 安装 PyTorch（约 2.4GB，最慢的一步）=="
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch --index-url https://download.pytorch.org/whl/cu124 \
  || pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "== 3/5 安装其余依赖 =="
pip install -r code/requirements-server.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "== 4/5 下载 bert-base-chinese（约 400MB）=="
export HF_HOME="$PWD/.hf-cache"
mkdir -p code/models
if python -c "
from huggingface_hub import snapshot_download
snapshot_download('bert-base-chinese', local_dir='code/models/bert-base-chinese',
                  allow_patterns=['*.json', '*.txt', '*.bin', '*.safetensors'])
"; then
  echo "   直连 HuggingFace 成功"
else
  echo "   直连失败，改用 hf-mirror 重试"
  export HF_ENDPOINT=https://hf-mirror.com
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('bert-base-chinese', local_dir='code/models/bert-base-chinese',
                  allow_patterns=['*.json', '*.txt', '*.bin', '*.safetensors'])
"
fi

echo "== 5/5 自检 =="
cd code
python -X utf8 -m scripts.doctor

echo
echo "环境就绪。下一步： bash run_train.sh 3e-5"
