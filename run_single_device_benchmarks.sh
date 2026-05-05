#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

"$PYTHON" benchmarks/single_device_benchmark.py \
  --model "${MODEL:-all}" \
  --dataset "${DATASET:-all}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --batches "${BATCHES:-20}" \
  --warmup-batches "${WARMUP_BATCHES:-3}" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-auto}" \
  ${DOWNLOAD:+--download} \
  ${AMP:+--amp}
