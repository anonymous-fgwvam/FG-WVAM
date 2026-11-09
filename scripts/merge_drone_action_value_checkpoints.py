#!/usr/bin/env python
"""Merge separately trained drone action and value FG-WVAM checkpoints.

The value-rerank inference path needs a single transformer containing:
  - video trunk weights,
  - action_* weights,
  - value_* weights.

For the usual setup, use the action checkpoint as the base and copy only
value_* tensors from the value checkpoint.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file, save_file


def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    p = Path(path)
    if p.is_dir():
        p = p / "diffusion_pytorch_model.safetensors"
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return load_file(str(p))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_checkpoint", required=True)
    parser.add_argument("--value_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--copy_prefix", action="append", default=["value_"])
    args = parser.parse_args()

    action_sd = load_safetensors(args.action_checkpoint)
    value_sd = load_safetensors(args.value_checkpoint)

    merged = dict(action_sd)
    copied = []
    skipped_shape = []
    for key, tensor in value_sd.items():
        if not any(key.startswith(prefix) for prefix in args.copy_prefix):
            continue
        if key in merged and merged[key].shape != tensor.shape:
            skipped_shape.append((key, list(tensor.shape), list(merged[key].shape)))
            continue
        merged[key] = tensor
        copied.append(key)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out / "diffusion_pytorch_model.safetensors"))
    summary = {
        "action_checkpoint": str(args.action_checkpoint),
        "value_checkpoint": str(args.value_checkpoint),
        "output": str(out / "diffusion_pytorch_model.safetensors"),
        "num_action_tensors": len(action_sd),
        "num_value_tensors": len(value_sd),
        "num_copied_value_tensors": len(copied),
        "skipped_shape": skipped_shape,
        "copy_prefix": args.copy_prefix,
    }
    (out / "merge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
