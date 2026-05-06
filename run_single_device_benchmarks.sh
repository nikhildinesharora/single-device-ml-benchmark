#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

ARGS=(benchmarks/single_device_benchmark.py)
if [ -n "${CONFIG:-}" ]; then
  ARGS+=(--config "$CONFIG")
  "$PYTHON" "${ARGS[@]}" ${DOWNLOAD:+--download} ${AMP:+--amp}
  exit 0
fi

"$PYTHON" "${ARGS[@]}" \
  --model "${MODEL:-all}" \
  --dataset "${DATASET:-all}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --batches "${BATCHES:-20}" \
  --epochs "${EPOCHS:-0}" \
  --batches-per-epoch "${BATCHES_PER_EPOCH:-0}" \
  --warmup-batches "${WARMUP_BATCHES:-3}" \
  --log-every "${LOG_EVERY:-100}" \
  --power-sample-interval "${POWER_SAMPLE_INTERVAL:-2.0}" \
  --lr "${LR:-0.01}" \
  --momentum "${MOMENTUM:-0.9}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-auto}" \
  ${DOWNLOAD:+--download} \
  ${AMP:+--amp}
