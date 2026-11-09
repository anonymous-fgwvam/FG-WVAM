#!/usr/bin/env bash
# Stage 2: add future target grounding and target-weighted video generation.
# Usage: bash scripts/train_stage2.sh [config]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-configs/paper/stage2_grounding_paper.yaml}"
GPUS="${GPUS:-1}"
PORT="${MASTER_PORT:-29602}"

cd "$REPO"
exec python -m torch.distributed.run \
  --standalone --nproc_per_node="$GPUS" --master_port="$PORT" \
  main.py --config_file "$CONFIG" --mode train \
  --runner_class_path runner/trainer.py --runner_class Trainer