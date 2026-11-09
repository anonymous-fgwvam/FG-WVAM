#!/usr/bin/env bash
# Stage 1: learn visual dynamics and the future target-heatmap head.
# Usage: bash scripts/train_stage1.sh [config]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-configs/paper/stage1_video_paper.yaml}"
GPUS="${GPUS:-1}"
PORT="${MASTER_PORT:-29601}"

cd "$REPO"
exec python -m torch.distributed.run \
  --standalone --nproc_per_node="$GPUS" --master_port="$PORT" \
  main.py --config_file "$CONFIG" --mode train \
  --runner_class_path runner/trainer.py --runner_class Trainer