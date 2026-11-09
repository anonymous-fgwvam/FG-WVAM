#!/usr/bin/env python
"""Small HTTP policy server for FG-WVAM drone policies.

The server keeps the heavy FG-WVAM/LTX models in the genie_envisioner Python
environment and exposes a tiny /act endpoint for AerialBench, which runs in a
different Python environment.
"""

import argparse
import base64
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from einops import rearrange
from yaml import Loader, load

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.inferencer import Inferencer
from data.drone_lerobot_dataset import (
    DRONE_ACTION_CATEGORICAL_NAMES,
    DRONE_ACTION_CATEGORICAL_VALUES,
)


def _prepare_stats(values: Optional[Sequence[float]], dim: int, is_std: bool) -> List[float]:
    fill = 1.0 if is_std else 0.0
    vals = [] if values is None else [float(v) for v in values]
    if len(vals) < dim:
        vals.extend([fill] * (dim - len(vals)))
    vals = vals[:dim]
    if is_std:
        vals = [v if abs(v) > 1e-8 else 1.0 for v in vals]
    return vals


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _decode_image(encoded: str, sample_size: Tuple[int, int]) -> np.ndarray:
    raw = base64.b64decode(encoded)
    img = Image.open(BytesIO(raw)).convert("RGB")
    # Dataset uses torchvision Resize(sample_size), whose tuple is (h, w).
    img = img.resize((int(sample_size[1]), int(sample_size[0])), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _normalise_frames(frames: Sequence[Sequence[str]], sample_size: Tuple[int, int]) -> torch.Tensor:
    """Decode payload frames to [v, c, t, h, w] float tensor in [-1, 1]."""
    if not frames:
        raise ValueError("payload.frames must contain at least one timestep")
    decoded_steps = []
    for step in frames:
        if len(step) != 4:
            raise ValueError(f"each timestep must contain 4 camera images, got {len(step)}")
        views = [_decode_image(item, sample_size) for item in step]
        decoded_steps.append(np.stack(views, axis=0))  # [v,h,w,3]
    arr = np.stack(decoded_steps, axis=1)  # [v,t,h,w,3]
    arr = arr.astype(np.float32) / 127.5 - 1.0
    arr = np.transpose(arr, (0, 4, 1, 2, 3))  # [v,c,t,h,w]
    return torch.from_numpy(np.ascontiguousarray(arr))


def _pad_or_trim_history(items: Sequence[Any], n: int) -> List[Any]:
    if not items:
        raise ValueError("history is empty")
    vals = list(items)[-n:]
    while len(vals) < n:
        vals.insert(0, vals[0])
    return vals


@dataclass
class PolicyResult:
    action: str
    raw_action: List[float]
    action_chunk: List[str]
    value_score: Optional[float]
    value_curve: Optional[List[float]]
    goal_heatmap: Optional[List[List[float]]] = None
    predicted_frames: Optional[List[List[str]]] = None
    goal_heatmaps: Optional[List[List[List[float]]]] = None


class FGDronePolicy:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mode = args.mode
        self.device = args.device
        self.weight_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32

        self.runner = Inferencer(
            args.config_file,
            output_dir=args.output_dir,
            weight_dtype=self.weight_dtype,
            device=args.device,
        )
        if args.checkpoint_path:
            self.runner.args.diffusion_model["model_path"] = args.checkpoint_path
        if args.num_inference_steps is not None:
            self.runner.args.num_inference_step = int(args.num_inference_steps)
        if args.disable_frame_ids_rope:
            self.runner.args.use_video_frame_ids_rope = False

        self.runner.prepare_models()
        self.pipe = self.runner.pipeline_class(
            self.runner.scheduler,
            self.runner.vae,
            self.runner.text_encoder,
            self.runner.tokenizer,
            self.runner.diffusion_model,
        )
        self.pipe.transformer.eval()

        data_cfg = self.runner.args.data.get("val", self.runner.args.data.get("train", {}))
        self.sample_size = tuple(int(x) for x in data_cfg.get("sample_size", [256, 256]))
        self.n_previous = int(data_cfg.get("n_previous", 2))
        self.chunk = int(data_cfg.get("chunk", 2))
        self.action_chunk = int(data_cfg.get("action_chunk", 2))
        self.value_dense = bool(data_cfg.get("value_dense", True))
        self.value_chunk = self.action_chunk if self.value_dense else self.chunk
        self.n_view = len(data_cfg.get("valid_cam", [0, 1, 2, 3]))
        self.action_dim = int(self.runner.args.diffusion_model["config"].get("action_in_channels", 14))
        self.value_dim = int(self.runner.args.diffusion_model["config"].get("value_in_channels", 1))
        self.use_start_video = bool(getattr(self.runner.args, "use_start_video", False))
        self.use_value_pose_condition = bool(getattr(self.runner.args, "use_value_pose_condition", False))
        self.use_frame_ids_rope = bool(getattr(self.runner.args, "use_video_frame_ids_rope", False))
        self.temporal_down_ratio = int(self.runner.TEMPORAL_DOWN_RATIO)

        stat_file = data_cfg.get("stat_file")
        self.stats = {}
        if stat_file and os.path.exists(stat_file):
            with open(stat_file, "r", encoding="utf-8") as f:
                self.stats = json.load(f)

        cfg_mean = self.runner.args.diffusion_model["config"].get("action_discrete_mean")
        cfg_std = self.runner.args.diffusion_model["config"].get("action_discrete_std")
        self.action_mean = _prepare_stats(cfg_mean, self.action_dim, is_std=False)
        self.action_std = _prepare_stats(cfg_std, self.action_dim, is_std=True)

    def _build_image_tensor(self, payload: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        frames = _pad_or_trim_history(payload.get("frames", []), self.n_previous)
        image = _normalise_frames(frames, self.sample_size)  # [v,c,t,h,w]
        _, _, t, h, w = image.shape
        image = image.unsqueeze(0)  # [b,v,c,t,h,w]
        image = rearrange(image, "b v c t h w -> (b v) c t h w")

        if self.use_start_video:
            start_frame = payload.get("start_frame") or frames[0]
            start = _normalise_frames([start_frame], self.sample_size).unsqueeze(0)
            start = rearrange(start, "b v c t h w -> (b v) c t h w")
            image = torch.cat([start, image], dim=2)

        return image.to(self.device, dtype=self.weight_dtype), None, h, w

    def _build_mem_frame_ids(self, payload: Dict[str, Any]) -> Optional[torch.Tensor]:
        if not self.use_frame_ids_rope:
            return None
        steps = payload.get("frame_steps") or payload.get("pose_steps") or [payload.get("step", 0)]
        steps = _pad_or_trim_history([float(s) for s in steps], self.n_previous)
        mem = torch.tensor([steps], device=self.device, dtype=torch.float32)
        if self.use_start_video:
            mem = torch.cat([torch.zeros(1, 1, device=self.device, dtype=torch.float32), mem], dim=1)
        return mem

    def _normalise_pose(self, pose: Sequence[float]) -> List[float]:
        raw = [float(v) for v in pose[:7]]
        if len(raw) < 7:
            raw.extend([0.0] * (7 - len(raw)))
        st = self.stats.get("drone_state_joint", {})
        mean = _prepare_stats(st.get("mean"), 14, is_std=False)[:7]
        std = _prepare_stats(st.get("std"), 14, is_std=True)[:7]
        norm7 = [(v - m) / s for v, m, s in zip(raw, mean, std)]
        return norm7 + [0.0] * 7

    def _build_pose_condition(self, payload: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_value_pose_condition:
            return None, None
        poses = payload.get("poses") or []
        if not poses:
            pose = payload.get("pose") or payload.get("start_pose") or [0.0] * 7
            poses = [pose]
        poses = _pad_or_trim_history(poses, self.n_previous)
        start_pose = payload.get("start_pose") or poses[0]

        pose_steps = payload.get("pose_steps") or payload.get("frame_steps") or [payload.get("step", 0)]
        pose_steps = _pad_or_trim_history([float(s) for s in pose_steps], self.n_previous)
        mode = getattr(self.runner.args, "value_pose_time_mode", "dataset")
        if mode == "relative_order":
            n_pose = 1 + len(poses)
            positions = [float(i) for i in range(1 - n_pose, 1)]
            denom = max(n_pose - 1, 1)
            pose_dt = [p / denom for p in positions]
        elif mode == "dataset":
            max_actions = max(float(payload.get("max_actions", 1) or 1), 1.0)
            positions = [0.0] + pose_steps
            pose_dt = [p / max_actions for p in positions]
        else:
            raise ValueError(
                f"Unsupported value_pose_time_mode={mode!r}; "
                "expected 'dataset' or 'relative_order'"
            )
        pose_rows = [self._normalise_pose(start_pose)] + [self._normalise_pose(p) for p in poses]
        pose_rows = [row + [dt] for row, dt in zip(pose_rows, pose_dt)]
        pose_state = torch.tensor([pose_rows], device=self.device, dtype=self.weight_dtype)
        pose_positions = torch.tensor([positions], device=self.device, dtype=torch.float32)
        return pose_state, pose_positions

    def _decode_action(self, action_row: Sequence[float]) -> str:
        raw = np.asarray(action_row[: self.action_dim], dtype=np.float32)
        mean = np.asarray(self.action_mean, dtype=np.float32)
        std = np.asarray(self.action_std, dtype=np.float32)
        raw = raw * std + mean
        prototypes = np.asarray(DRONE_ACTION_CATEGORICAL_VALUES, dtype=np.float32)
        dists = np.abs(raw[: prototypes.shape[1]][None, :] - prototypes).sum(axis=1)
        return str(DRONE_ACTION_CATEGORICAL_NAMES[int(dists.argmin())])

    def act(self, payload: Dict[str, Any]) -> PolicyResult:
        prompt = str(payload.get("prompt") or payload.get("description") or "").strip()
        if not prompt:
            raise ValueError("payload.prompt is required")
        if not prompt.lower().startswith("navigate to"):
            prompt = f"navigate to the {prompt}"

        image, _, h, w = self._build_image_tensor(payload)
        mem_frame_ids = self._build_mem_frame_ids(payload)
        value_pose_state, value_pose_positions = self._build_pose_condition(payload)

        _has_value_expert = bool(
            self.runner.args.diffusion_model["config"].get("value_expert", False)
        )
        return_value = (
            self.mode == "value_rerank"
            or (getattr(self.args, "return_value_in_action", False) and _has_value_expert)
        )
        explore_config = {
            "explore_steps": int(payload.get("explore_steps", self.args.explore_steps)),
            "dynamic_groups": int(payload.get("dynamic_groups", self.args.dynamic_groups)),
            "value_groups": int(payload.get("value_groups", self.args.value_groups)),
            "sigma_decay": float(payload.get("sigma_decay", self.args.sigma_decay)),
            "alpha_smooth": float(payload.get("alpha_smooth", self.args.alpha_smooth)),
            "value_elites": float(payload.get("value_elites", self.args.value_elites)),
            "dynamic_elites": float(payload.get("dynamic_elites", self.args.dynamic_elites)),
            "selection_mode": str(payload.get("selection_mode", self.args.selection_mode)),
        }
        if explore_config["selection_mode"] == "argmax":
            # Selection mode: K candidate futures, one value score each, argmax.
            explore_config["dynamic_groups"] = int(payload.get("num_candidates", self.args.num_candidates))
            explore_config["value_groups"] = int(payload.get("value_samples", 1))

        with torch.no_grad():
            preds = self.pipe.infer(
                image=image,
                prompt=[prompt],
                negative_prompt="",
                num_inference_steps=int(self.runner.args.num_inference_step),
                decode_timestep=0.03,
                decode_noise_scale=0.025,
                guidance_scale=float(self.args.guidance_scale),
                height=h,
                width=w,
                n_view=self.n_view,
                return_action=True,
                return_value=return_value,
                value_chunk=self.value_chunk,
                n_prev=self.n_previous + (1 if self.use_start_video else 0),
                chunk=(self.chunk - 1) // self.temporal_down_ratio + 1,
                return_video=bool(getattr(self.args, "return_video", False)),
                exploration=(self.mode == "value_rerank"),
                selection_mode=self.args.selection_mode,
                explore_config=explore_config,
                noise_seed=int(payload.get("seed", self.args.seed)),
                action_chunk=self.action_chunk,
                history_action_state=None,
                mem_frame_ids=mem_frame_ids,
                use_video_frame_ids_rope=self.use_frame_ids_rope,
                value_pose_state=value_pose_state,
                value_pose_positions=value_pose_positions,
                pixel_wise_timestep=bool(self.runner.args.pixel_wise_timestep),
                n_chunk=1,
                action_dim=self.action_dim,
                value_dim=self.value_dim,
            )[0]

        actions = preds["action"][0].detach().float().cpu().numpy()
        action_chunk = [self._decode_action(row) for row in actions]
        value_curve = None
        value_score = None
        if return_value and "value" in preds:
            values = preds["value"][0].detach().float().cpu().numpy()
            curve = values[:, -1].astype(float).tolist()
            value_curve = [float(v) for v in curve]
            value_score = float(np.mean(curve)) if curve else None

        goal_heatmap = None
        goal_heatmaps = None
        if getattr(preds, "get", None) is not None:
            hm = preds.get("goal_heatmap")
            if hm is None:
                hm = preds.get("goal_latent_heatmap")
            if hm is not None:
                hm_np = hm.detach().float().cpu().numpy().astype(float)
                if hm_np.ndim == 4 and hm_np.shape[0] >= self.n_view:
                    # (b*v, t, h, w) with b=1
                    goal_heatmap = hm_np[0].tolist()
                    goal_heatmaps = [hm_np[i].tolist()
                                     for i in range(min(self.n_view, hm_np.shape[0]))]
                else:
                    goal_heatmap = hm_np.tolist()
                    goal_heatmaps = [hm_np.tolist()]

        predicted_frames = None
        if getattr(self.args, "return_video", False) and "video" in preds:
            import cv2
            video = preds["video"]
            if torch.is_tensor(video):
                video = video.detach().float().cpu().numpy()
            # (b*v, c, t, h, w) with b=1
            views = min(self.n_view, video.shape[0])
            predicted_frames = []
            for vi in range(views):
                view_frames = []
                for ti in range(video.shape[2]):
                    arr = ((video[vi, :, ti] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    arr = np.transpose(arr, (1, 2, 0))
                    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                    if not ok:
                        raise RuntimeError("failed to encode predicted frame")
                    view_frames.append(base64.b64encode(buf.tobytes()).decode("ascii"))
                predicted_frames.append(view_frames)

        return PolicyResult(
            action=action_chunk[0],
            raw_action=[float(v) for v in actions[0].tolist()],
            action_chunk=action_chunk,
            value_score=value_score,
            value_curve=value_curve,
            goal_heatmap=goal_heatmap,
            predicted_frames=predicted_frames,
            goal_heatmaps=goal_heatmaps,
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("FG_POLICY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FG_POLICY_PORT", "18081")))
    parser.add_argument("--mode", choices=("action", "value_rerank"), default=os.environ.get("FG_POLICY_MODE", "action"))
    parser.add_argument("--config_file", default=os.environ.get("FG_POLICY_CONFIG", "configs/paper/stage3_action_value_paper.yaml"))
    parser.add_argument("--checkpoint_path", default=os.environ.get("FG_POLICY_CHECKPOINT", ""))
    parser.add_argument("--output_dir", default=os.environ.get("FG_POLICY_OUTPUT_DIR", "experiments/fgwvam_policy_server"))
    parser.add_argument("--device", default=os.environ.get("FG_POLICY_DEVICE", "cuda:0"))
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default=os.environ.get("FG_POLICY_DTYPE", "bf16"))
    parser.add_argument("--return_value_in_action", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=int(os.environ["FG_POLICY_NUM_STEPS"]) if os.environ.get("FG_POLICY_NUM_STEPS") else None)
    parser.add_argument("--guidance_scale", type=float, default=float(os.environ.get("FG_POLICY_GUIDANCE", "1.0")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("FG_POLICY_SEED", "42")))
    parser.add_argument("--disable_frame_ids_rope", action="store_true")
    parser.add_argument("--explore_steps", type=int, default=int(os.environ.get("FG_VALUE_EXPLORE_STEPS", "1")))
    parser.add_argument("--dynamic_groups", type=int, default=int(os.environ.get("FG_VALUE_DYNAMIC_GROUPS", "8")))
    parser.add_argument("--value_groups", type=int, default=int(os.environ.get("FG_VALUE_GROUPS", "3")))
    parser.add_argument("--sigma_decay", type=float, default=float(os.environ.get("FG_VALUE_SIGMA_DECAY", "0.5")))
    parser.add_argument("--alpha_smooth", type=float, default=float(os.environ.get("FG_VALUE_ALPHA_SMOOTH", "0.9")))
    parser.add_argument("--value_elites", type=float, default=float(os.environ.get("FG_VALUE_ELITES", "0.2")))
    parser.add_argument("--dynamic_elites", type=float, default=float(os.environ.get("FG_VALUE_DYNAMIC_ELITES", "0.3")))
    parser.add_argument("--selection_mode", choices=("elite", "argmax"), default=os.environ.get("FG_SELECTION_MODE", "argmax"))
    parser.add_argument("--return_video", action="store_true", help="also return decoded predicted future frames")
    parser.add_argument("--num_candidates", type=int, default=int(os.environ.get("FG_NUM_CANDIDATES", "3")))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)
    policy = FGDronePolicy(args)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *msg_args: Any) -> None:
            print("[http] " + fmt % msg_args, flush=True)

        def do_GET(self) -> None:
            if self.path == "/health":
                _json_response(self, 200, {"ok": True, "mode": policy.mode})
            else:
                _json_response(self, 404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/act":
                _json_response(self, 404, {"ok": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = policy.act(payload)
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "mode": policy.mode,
                        "action": result.action,
                        "raw_action": result.raw_action,
                        "action_chunk": result.action_chunk,
                        "value_score": result.value_score,
                        "value_curve": result.value_curve,
                        "goal_heatmap": result.goal_heatmap,
                        "predicted_frames": result.predicted_frames,
                        "goal_heatmaps": result.goal_heatmaps,
                    },
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                _json_response(self, 500, {"ok": False, "error": repr(exc)})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[server] FG-WVAM drone policy ready on http://{args.host}:{args.port} mode={args.mode}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
