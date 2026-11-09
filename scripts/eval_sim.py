#!/usr/bin/env python3
"""Closed-loop simulator evaluation for a served FG-WVAM policy.

The simulator is not part of this repository. Provide an AerialDojo
(ProjectAirSim) checkout with ``--sim-repo`` and a task file with ``--task-file``.
The script asks the policy server for one action per step, executes it in the
simulator, and reports success rate (SR), oracle success rate (OSR), and
distance-to-stop (DTS).

The number of episodes is determined by the task file. ``--episodes 0``
evaluates every task; set ``--start-episode`` / ``--episodes`` to evaluate a
slice. The script does not hardcode 300 episodes: that count depends on the
benchmark split you provide.

Example:
    python scripts/eval_sim.py \
      --sim-repo /path/to/AerialDojo \
      --task-file /path/to/tasks.json \
      --policy-url http://127.0.0.1:18099 \
      --seeds 42 43 44 \
      --success-distance 3.0 \
      --max-actions 150 \
      --output-jsonl eval_aerialdojo.jsonl
"""
import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

EXPECTED_CAMERAS = ("0", "1", "2", "3")


def _load_simulator(sim_repo: str):
    if sim_repo:
        sys.path.insert(0, sim_repo)
    try:
        from AerialDojo.env_uav import SimpleUAVEnv, UAVEnvConfig
        from AerialDojo.env_utils_uav import ActionSettings
    except ImportError as exc:
        raise SystemExit(
            "AerialDojo simulator not found. Pass --sim-repo /path/to/AerialDojo "
            "(the directory containing the AerialDojo package)."
        ) from exc
    return SimpleUAVEnv, UAVEnvConfig, ActionSettings


def _to_rgb_images(observation: Dict[str, Any]) -> List[np.ndarray]:
    import cv2

    images = observation.get("rgb")
    if images is None or len(images) != 4:
        raise RuntimeError("need 4 RGB camera images")
    out = []
    for image in images:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise RuntimeError("bad image shape: %s" % (arr.shape,))
        # AerialBench cameras return BGR; the policy is trained on RGB.
        out.append(cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2RGB))
    return out


def _encode_png_rgb(image: np.ndarray) -> str:
    import cv2

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class FGRemotePolicy:
    def __init__(self, policy_url: str, history: int = 2, timeout: float = 600.0,
                 seed: int = 42, max_actions: int = 150) -> None:
        self.policy_url = policy_url.rstrip("/")
        self.history = int(history)
        self.timeout = float(timeout)
        self.seed = int(seed)
        self.max_actions = int(max_actions)
        self.start_frame: Optional[List[np.ndarray]] = None
        self.start_pose: Optional[List[float]] = None
        self.frames: List[List[np.ndarray]] = []
        self.poses: List[List[float]] = []
        self.steps: List[int] = []

    def reset(self, observation: Dict[str, Any]) -> None:
        self.start_frame = None
        self.start_pose = None
        self.frames = []
        self.poses = []
        self.steps = []

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.policy_url + "/act", data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError("policy HTTP %s: %s" % (exc.code, body)) from exc
        if not result.get("ok", False):
            raise RuntimeError("policy error: %s" % (result,))
        return result

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        images = _to_rgb_images(observation)
        pose = [float(v) for v in observation.get("pose", [])[:7]]
        if len(pose) < 7:
            pose = pose + [0.0] * (7 - len(pose))
        step = int(observation.get("step", 0))

        if self.start_frame is None:
            self.start_frame = [np.asarray(img).copy() for img in images]
        if self.start_pose is None:
            self.start_pose = list(pose)

        self.frames.append([np.asarray(img).copy() for img in images])
        self.poses.append(list(pose))
        self.steps.append(step)
        self.frames = self.frames[-self.history:]
        self.poses = self.poses[-self.history:]
        self.steps = self.steps[-self.history:]

        description = str(observation.get("task", {}).get("description", "")).strip()
        if not description:
            raise RuntimeError("task.description is empty")

        payload = {
            "description": description,
            "frames": [[_encode_png_rgb(v) for v in frame] for frame in self.frames],
            "frame_steps": list(self.steps),
            "start_frame": [_encode_png_rgb(v) for v in (self.start_frame or [])],
            "pose": pose,
            "poses": self.poses,
            "pose_steps": list(self.steps),
            "start_pose": self.start_pose,
            "step": step,
            "max_actions": self.max_actions,
            "seed": self.seed,
        }
        result = self._post(payload)
        return {
            "action": str(result["action"]),
            "step_size": None,
            "raw_action": [float(v) for v in result.get("raw_action", [])],
            "action_chunk": [str(v) for v in result.get("action_chunk", [])],
            "value_score": result.get("value_score"),
            "value_curve": result.get("value_curve"),
            "goal_heatmap": result.get("goal_heatmap"),
            "policy_mode": result.get("mode"),
            "rgb_images": images,
        }


def save_observation_frames(frame_dir: Path, seed: int, episode_index: int,
                            step_index: int, decision: Dict[str, Any],
                            observation: Dict[str, Any]) -> None:
    import cv2

    ep_dir = frame_dir / ("seed_%d" % seed) / ("episode_%03d" % episode_index)
    ep_dir.mkdir(parents=True, exist_ok=True)
    for vi in range(min(4, len(decision["rgb_images"]))):
        view = np.asarray(decision["rgb_images"][vi])
        out = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(ep_dir / ("step_%04d_cam%d.png" % (step_index, vi))), out)
    record = {
        "seed": seed,
        "episode_index": episode_index,
        "step_index": step_index,
        "action": decision["action"],
        "raw_action": decision["raw_action"],
        "action_chunk": decision["action_chunk"],
        "value_score": decision.get("value_score"),
        "task_id": observation.get("task", {}).get("task_id"),
        "description": observation.get("task", {}).get("description"),
        "distance_to_goal": observation.get("distance_to_goal"),
        "success": observation.get("success"),
        "collision": observation.get("collision"),
    }
    with open(ep_dir / "steps.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop AerialDojo evaluation for FG-WVAM.")
    parser.add_argument("--sim-repo", default="",
                        help="directory containing the AerialDojo package")
    parser.add_argument("--task-file", required=True,
                        help="simulator task file; determines the episode count")
    parser.add_argument("--policy-url", required=True)
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=35002)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=150)
    parser.add_argument("--success-distance", type=float, default=3.0,
                        help="success radius in meters (UAV-ON and AerialDojo differ)")
    parser.add_argument("--motion-mode", choices=("command", "teleport"), default="teleport")
    parser.add_argument("--scene-config-name",
                        default="scene_aerialdojo_drone_nonphysics_4cam_640.jsonc")
    parser.add_argument("--episodes", type=int, default=0,
                        help="0 evaluates every task in --task-file")
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--save-frames-dir", default="")
    return parser.parse_args(argv)


def evaluate_seed(args, seed, env, policy_cls, max_actions):
    policy = policy_cls(policy_url=args.policy_url, history=args.history,
                        timeout=args.request_timeout, seed=seed,
                        max_actions=max_actions)
    save_root = Path(args.save_frames_dir).expanduser().resolve() if args.save_frames_dir else None
    start = max(0, int(args.start_episode))
    remaining = len(env.tasks) - start
    count = remaining if args.episodes <= 0 else min(int(args.episodes), remaining)
    summaries = []

    for episode_index in range(start, start + count):
        observation = env.reset(indices=[episode_index])[0]
        policy.reset(observation)
        total_reward = 0.0
        stopped_by_policy = False
        last_info = {
            "success": observation["success"],
            "collision": observation["collision"],
            "within_success_radius": observation["within_success_radius"],
            "oracle_success": observation["oracle_success"],
            "terminated": observation["terminated"],
            "truncated": observation["truncated"],
            "termination_reason": observation["termination_reason"],
            "distance_to_goal": observation["distance_to_goal"],
            "move_distance": observation["move_distance"],
        }
        frame_step = 0
        while not observation["done"]:
            decision = policy.act(observation)
            action = decision["action"]
            stopped_by_policy = action == "stop"
            if save_root is not None:
                save_observation_frames(save_root, seed, episode_index, frame_step,
                                        decision, observation)
            frame_step += 1
            observations, rewards, terminated, truncated, infos = env.step_gymnasium(
                [action], step_sizes=[decision["step_size"]], is_fixed=True)
            observation = observations[0]
            total_reward += float(rewards[0])
            last_info = infos[0]
            if terminated[0] or truncated[0]:
                break

        summary = {
            "seed": seed,
            "episode_index": episode_index,
            "task_id": observation["task"]["task_id"],
            "steps": observation["step"],
            "success": bool(last_info["success"]),
            "collision": bool(last_info["collision"]),
            "within_success_radius": bool(last_info["within_success_radius"]),
            "oracle_success": bool(last_info["oracle_success"]),
            "terminated": bool(last_info["terminated"]),
            "truncated": bool(last_info["truncated"]),
            "termination_reason": str(last_info["termination_reason"]),
            "stopped_by_policy": bool(stopped_by_policy),
            "distance_to_goal_m": float(last_info["distance_to_goal"]),
            "move_distance_m": float(last_info["move_distance"]),
            "total_reward": total_reward,
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summaries


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    SimpleUAVEnv, UAVEnvConfig, ActionSettings = _load_simulator(args.sim_repo)

    action_settings = ActionSettings(
        forward_step=1.0, lateral_step=1.0, vertical_step=1.0, turn_degrees=15.0)
    config = UAVEnvConfig(
        server_host=args.server_host,
        server_port=args.server_port,
        gpu_id=args.gpu_id,
        batch_size=1,
        max_steps=args.max_actions,
        success_distance=args.success_distance,
        cameras=EXPECTED_CAMERAS,
        motion_mode=args.motion_mode,
        action_settings=action_settings,
        rgb_mode="raw",
        depth_mode="uint8",
        reuse_scenes=True,
        max_scene_reuses=5000,
        terminate_on_success=True,
        scene_config_name=args.scene_config_name,
    )
    env = SimpleUAVEnv(task_file=args.task_file, config=config)

    all_summaries = []
    try:
        for seed in args.seeds:
            summaries = evaluate_seed(args, seed, env, FGRemotePolicy, args.max_actions)
            all_summaries.extend(summaries)
            n = max(len(summaries), 1)
            sr = 100.0 * sum(s["success"] for s in summaries) / n
            osr = 100.0 * sum(s["oracle_success"] for s in summaries) / n
            dts = sum(s["distance_to_goal_m"] for s in summaries) / n
            print("SEED %d: episodes=%d SR=%.1f%% OSR=%.1f%% DTS=%.2fm" % (
                seed, len(summaries), sr, osr, dts), flush=True)
    finally:
        env.close()

    if args.output_jsonl:
        path = Path(args.output_jsonl).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with open(path, mode, encoding="utf-8") as f:
            for item in all_summaries:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    n = max(len(all_summaries), 1)
    sr = 100.0 * sum(s["success"] for s in all_summaries) / n
    osr = 100.0 * sum(s["oracle_success"] for s in all_summaries) / n
    dts = sum(s["distance_to_goal_m"] for s in all_summaries) / n
    print("OVERALL: episodes=%d SR=%.1f%% OSR=%.1f%% DTS=%.2fm" % (
        len(all_summaries), sr, osr, dts), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())