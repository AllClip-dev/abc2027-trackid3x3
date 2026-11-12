#!/usr/bin/env bash
# 単一ラッパー: 学習 → 終了状態の更新．status.json / command.txt / env.txt / run.log を <out_dir> に残す．
#   nohup bash run_train.sh <dataset_dir> <out_dir> [train_rfdetr.py の追加引数...] &
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ds=$1; out=$2; shift 2
PY=${PYTHON:-python}  # rfdetr 1.6.5 と pytorch-lightning の入った環境の python
# 学習用の依存を別ディレクトリに入れた場合は EXTRA_DEPS で PYTHONPATH に足す
[ -n "${EXTRA_DEPS:-}" ] && export PYTHONPATH=$EXTRA_DEPS
mkdir -p "$out"
echo "python train_rfdetr.py --dataset-dir $ds --output-dir $out $*" > "$out/command.txt"
"$PY" - > "$out/env.txt" 2>&1 <<PY
import importlib.metadata as m, torch
print("python", __import__("sys").version.split()[0]); print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0)); print("rfdetr", m.version("rfdetr"))
for p in ("pytorch-lightning", "torchmetrics", "pycocotools", "albumentations", "faster-coco-eval"): print(p, m.version(p))
PY
echo "{\"state\": \"running\", \"pid\": $$, \"started\": \"$(date +%FT%T)\", \"gpu\": \"0\", \"out_dir\": \"$out\"}" > "$out/status.json"
CUDA_VISIBLE_DEVICES=0 "$PY" "$HERE/train_rfdetr.py" --dataset-dir "$ds" --output-dir "$out" "$@" > "$out/run.log" 2>&1
rc=$?
state=succeeded; [ $rc -ne 0 ] && state=failed
echo "{\"state\": \"$state\", \"exit_code\": $rc, \"finished\": \"$(date +%FT%T)\", \"out_dir\": \"$out\"}" > "$out/status.json"
