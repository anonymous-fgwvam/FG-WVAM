#!/usr/bin/env bash
# Stage 2b: Future Value Expert pretraining on 1:1 successful/failed rollouts.
# Usage: bash scripts/train_stage2b.sh [config]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-configs/paper/stage2b_value_pretrain.yaml}"
GPUS="${GPUS:-1}"
PORT="${MASTER_PORT:-29604}"

cd "$REPO"
exec python -m torch.distributed.run \
  --standalone --nproc_per_node="$GPUS" --master_port="$PORT" \
  main.py --config_file "$CONFIG" --mode train \
  --runner_class_path runner/trainer.py --runner_class Trainer