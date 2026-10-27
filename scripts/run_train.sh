#!/usr/bin/env bash
# Train residual healthy-baseline v7 (full-scale residual + detail loss).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-$HOME/miniconda3/envs/torch/bin/python}"
NPROC="${NPROC:-2}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29507}"

OUT="${OUT:-runs/fundus2rnflt_v7_healthy_baseline}"
LOG="${LOG:-$OUT/train.log}"
RESUME="${RESUME:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"

RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
fi

mkdir -p "$(dirname "$LOG")"
echo "Launching healthy-baseline v7 DDP nproc=$NPROC GPUs=$CUDA_VISIBLE_DEVICES -> $OUT"
echo "Log: $LOG"

export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

nohup "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC" \
  -m rnfltnorm.train \
  --output-dir "$OUT" \
  --batch-size "$BATCH_SIZE" \
  --max-epochs 120 \
  --backbone-lr 3e-5 \
  --decoder-lr 3e-4 \
  --warmup-epochs 5 \
  --num-workers 4 \
  "${RESUME_ARGS[@]}" \
  >> "$LOG" 2>&1 &

echo "PID $!"
echo "tail -f $LOG"
