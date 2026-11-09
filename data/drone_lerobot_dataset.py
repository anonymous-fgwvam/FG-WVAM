import sys
import os
import json
import random
import traceback
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
from moviepy.editor import VideoFileClip
import pandas as pd

from utils import zero_rank_print


# Per-dimension discrete mode kept for older drone action experiments.
DRONE_ACTION_DISCRETE_MAP = {
    0: [0.0, 0.3],
    2: [-0.3, 0.0],
    3: [0.0, 0.5],
}
DRONE_ACTIVE_DIMS = sorted(DRONE_ACTION_DISCRETE_MAP.keys())  # [0, 2, 3]
DRONE_NUM_ACTIVE_DIMS = len(DRONE_ACTIVE_DIMS)  # 3
DRONE_NUM_CLASSES_PER_DIM = 2

# Normalization stats (from drone_lerobot_stats.json, drone_joint)
DRONE_ACTION_MEAN = [-0.006254877492078516, 0.03775736988980217, -0.046845550813993025, -0.005772655421493887, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DRONE_ACTION_STD = [0.20451637359693073, 0.09950774568110587, 0.10890077121909703, 0.25133892058388857, 1e-06, 1e-06, 1e-06, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

DRONE_ACTION_CATEGORICAL_NAMES = ("stop", "forward", "ascend", "descend", "rotl", "rotr")
DRONE_ACTION_CATEGORICAL_VALUES = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
)
DRONE_NUM_ACTION_CLASSES = len(DRONE_ACTION_CATEGORICAL_VALUES)


def _prepare_action_stats(values, defaults, action_dim=14, is_std=False):
    vals = list(defaults if values is None else values)
    fill = 1.0 if is_std else 0.0
    if len(vals) < action_dim:
        vals = vals + [fill] * (action_dim - len(vals))
    vals = vals[:action_dim]
    if is_std:
        vals = [v if abs(float(v)) > 1e-8 else 1.0 for v in vals]
    return vals


def _normalize_action_values(raw_values, mean=None, std=None, action_dim=14):
    mean_vals = _prepare_action_stats(mean, DRONE_ACTION_MEAN, action_dim=action_dim, is_std=False)
    std_vals = _prepare_action_stats(std, DRONE_ACTION_STD, action_dim=action_dim, is_std=True)
    normed_values = []
    for raw in raw_values:
        vals = list(raw)
        if len(vals) < action_dim:
            vals = vals + [0.0] * (action_dim - len(vals))
        vals = vals[:action_dim]
        normed_values.append([(v - m) / s for v, m, s in zip(vals, mean_vals, std_vals)])
    return normed_values


def get_discrete_action_value_map(device, dtype, mode="per_dim", mean=None, std=None, action_dim=14):
    """Return class_id→normalized_value lookup for pipeline inference.

    Returns:
        per_dim: torch.Tensor of shape (num_active_dims, num_classes_per_dim)
        categorical: torch.Tensor of shape (num_classes, action_dim)
    """
    if mode == "categorical":
        vals = _normalize_action_values(
            DRONE_ACTION_CATEGORICAL_VALUES,
            mean=mean,
            std=std,
            action_dim=action_dim,
        )
        return torch.tensor(vals, device=device, dtype=dtype)

    if mode != "per_dim":
        raise ValueError(f"Unknown action_discrete_mode: {mode}")

    mean_vals = _prepare_action_stats(mean, DRONE_ACTION_MEAN, action_dim=action_dim, is_std=False)
    std_vals = _prepare_action_stats(std, DRONE_ACTION_STD, action_dim=action_dim, is_std=True)
    vals = []
    for dim in DRONE_ACTIVE_DIMS:
        raw_vals = DRONE_ACTION_DISCRETE_MAP[dim]
        if len(raw_vals) == 1:
            normed = [(raw_vals[0] - mean_vals[dim]) / std_vals[dim]] * 2
        else:
            normed = [(v - mean_vals[dim]) / std_vals[dim] for v in raw_vals]
        vals.append(normed)
    vals = torch.tensor(vals, device=device, dtype=dtype)
    return vals


def load_jsonl(jsonl_path):
    """Load newline-delimited JSON file."""
    data = []
    with open(jsonl_path, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class DroneLeRobotDataset(Dataset):
    """Dataset for drone data in LeRobot format.

    Expected structure:
        DATA_ROOT/
        ├── data/
        │   └── chunk-XXX/
        │       └── episode_XXXXXX.parquet   (action: T*7, observation.pose: T*7)
        ├── videos/
        │   └── chunk-XXX/
        │       ├── cam_front/
        │       │   └── episode_XXXXXX.mp4
        │       ├── cam_left/
        │       ├── cam_right/
        │       └── cam_down/
        └── meta/
            ├── episodes.jsonl   (episode_index, tasks, length)
            ├── tasks.jsonl      (task_index, task)
            └── info.json

    Key adaptation from lerobot_like_dataset.py:
    - 7-dim action → extended to 14-dim (zero-padded)
    - 7-dim observation.pose → extended to 14-dim (zero-padded)
    - Camera names: cam_front, cam_left, cam_right, cam_down
    - Captions loaded from tasks.jsonl
    """

    def __init__(
        self,
        data_roots,
        domains,
        sample_size=(256, 256),
        sample_n_frames=64,
        preprocess="resize",
        valid_cam=("cam_front", "cam_left", "cam_right", "cam_down"),
        chunk=1,
        action_chunk=None,
        n_previous=-1,
        previous_pick_mode="uniform",
        random_crop=False,
        dataset_info_cache_path=None,
        action_type="absolute",
        action_space="joint",
        ignore_seek=False,
        use_unified_prompt=False,
        unified_prompt="best quality, consistent and smooth motion, realistic, clear and distinct.",
        fix_epiidx=None,
        fix_sidx=None,
        fix_mem_idx=None,
        stat_file=None,
        action_key="action",
        state_key="observation.pose",
        min_length=0,
        value_key="state_value",
        value_dense=True,
        grounding_key=None,
        grounding_dense=True,
        grounding_dim=1,
        grounding_weight_key=None,
        grounding_weight_dense=True,
        noisy_video=None,
        action_discrete_mode="per_dim",
        add_start_video=False,
        split=None,
        goal_label_dir=None,
        goal_label_cache_size=32,
        window_end=False,
    ):
        assert action_type in ("absolute", "delta", "relative")
        self.action_type = action_type
        assert action_space in ("eef", "joint")
        self.action_space = action_space
        if action_discrete_mode not in ("per_dim", "categorical"):
            raise ValueError(f"Unknown action_discrete_mode: {action_discrete_mode}")
        self.action_discrete_mode = action_discrete_mode
        self.split = split

        self.random_crop = random_crop
        if not isinstance(valid_cam, (list, tuple)):
            valid_cam = [valid_cam]
        self.valid_cam = list(valid_cam)
        self.value_key = value_key
        self.value_dense = value_dense
        self.grounding_key = grounding_key
        self.grounding_dense = grounding_dense
        self.grounding_dim = grounding_dim
        self.grounding_weight_key = grounding_weight_key
        self.grounding_weight_dense = grounding_weight_dense
        self.goal_label_dir = goal_label_dir
        self.goal_label_cache_size = max(int(goal_label_cache_size), 0)
        self.window_end = bool(window_end)
        self._label_cache = OrderedDict()

        if len(data_roots) == 1 and len(domains) > 1:
            data_roots = data_roots * len(domains)
        self.data_roots = data_roots

        self.action_key = action_key
        self.state_key = state_key
        self.min_length = min_length

        # Build per-episode info by scanning meta files
        self.dataset = []
        self._load_episodes_from_meta()

        if dataset_info_cache_path is not None and os.path.exists(dataset_info_cache_path):
            zero_rank_print(f"Load Cache Dataset Information from {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "r") as f:
                self.dataset = json.load(f)
        elif dataset_info_cache_path is not None:
            zero_rank_print(f"Save Cache Dataset Information to {dataset_info_cache_path}")
            os.makedirs(os.path.dirname(dataset_info_cache_path), exist_ok=True)
            with open(dataset_info_cache_path, "w") as f:
                json.dump(self.dataset, f)

        if self.min_length > 0:
            before = len(self.dataset)
            self.dataset = [ep for ep in self.dataset if ep[6] >= self.min_length]
            zero_rank_print(f"Filtered episodes by min_length={self.min_length}: {before} -> {len(self.dataset)}")

        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.chunk = chunk
        if action_chunk is None:
            action_chunk = chunk
        self.action_chunk = action_chunk
        self.video_temporal_stride = self.action_chunk // self.chunk
        assert self.chunk * self.video_temporal_stride == self.action_chunk

        self.sample_n_frames = sample_n_frames
        self.sample_size = sample_size

        if preprocess == "center_crop_resize":
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(sample_size)),
                transforms.CenterCrop(sample_size),
            ])
        elif preprocess == "resize":
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        else:
            raise NotImplementedError(f"preprocess '{preprocess}' not supported")

        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.preprocess = preprocess

        if n_previous > 1:
            self.n_previous = int(n_previous)
            self.previous_pick_mode = previous_pick_mode
        else:
            self.n_previous = int(self.sample_n_frames - self.chunk)
            self.previous_pick_mode = "uniform"

        self.use_unified_prompt = use_unified_prompt
        self.unified_prompt = unified_prompt

        # Validation only
        self.fix_epiidx = fix_epiidx
        self.fix_sidx = fix_sidx
        self.fix_mem_idx = fix_mem_idx
        self._add_start_video = add_start_video
        self.ignore_seek = ignore_seek

        # Load action statistics
        self.StatisticInfo = {
            "drone_joint": {"mean": [0.0]*14, "std": [1.0]*14},
            "drone_delta_joint": {"mean": [0.0]*14, "std": [1.0]*14},
            "drone_state_joint": {"mean": [0.0]*14, "std": [1.0]*14},
        }
        if stat_file is not None and os.path.exists(stat_file):
            with open(stat_file, "r") as f:
                loaded = json.load(f)
            # Merge loaded stats into defaults (handles both 7-dim and 14-dim)
            for domain_key in loaded:
                if "joint" in domain_key:
                    mean = loaded[domain_key]["mean"]
                    std = loaded[domain_key]["std"]
                    # Extend to 14-dim if 7-dim
                    if len(mean) == 7:
                        mean = list(mean) + [0.0] * 7
                        std = list(std) + [1.0] * 7
                    std = [s if abs(float(s)) > 1e-8 else 1.0 for s in std]
                    self.StatisticInfo[domain_key] = {"mean": mean, "std": std}

    def _load_episodes_from_meta(self):
        """Build episode list from meta/episodes.jsonl + meta/tasks.jsonl."""
        for data_root in self.data_roots:
            meta_folder = os.path.join(data_root, "meta")
            domain_name = "drone"

            # Load task descriptions
            tasks_jsonl = os.path.join(meta_folder, "tasks.jsonl")
            task_map = {}
            if os.path.exists(tasks_jsonl):
                for task_item in load_jsonl(tasks_jsonl):
                    task_map[task_item["task_index"]] = task_item["task"]
            else:
                # Also try tasks.json (README shows both)
                tasks_json = os.path.join(meta_folder, "tasks.json")
                if os.path.exists(tasks_json):
                    raw = load_jsonl(tasks_json)
                    for item in raw:
                        if isinstance(item, dict) and "task_index" in item:
                            task_map[item["task_index"]] = item["task"]
                        elif isinstance(item, dict) and "task" in item:
                            task_map[item.get("task_index", 0)] = item["task"]

            # Load episodes
            episodes_jsonl = os.path.join(meta_folder, "episodes.jsonl")
            epiosdes_data = load_jsonl(episodes_jsonl)

            # Load info for chunks_size
            info_json = os.path.join(meta_folder, "info.json")
            chunks_size = 1000
            metainfo = {}
            if os.path.exists(info_json):
                with open(info_json, "r") as f:
                    metainfo = json.load(f)
                chunks_size = metainfo.get("chunks_size", 1000)

            split_range = None
            if self.split is not None:
                splits = metainfo.get("splits", {})
                if self.split not in splits:
                    raise ValueError(f"Split '{self.split}' not found in {info_json}")
                split_start, split_end = splits[self.split].split(":")
                split_range = (int(split_start), int(split_end))

            for episode_data in epiosdes_data:
                episode_index = episode_data["episode_index"]
                if split_range is not None and not (split_range[0] <= episode_index < split_range[1]):
                    continue

                tasks = episode_data["tasks"]
                if len(tasks) > 1:
                    caption = random.choice(tasks)
                else:
                    caption = tasks[0] if tasks else ""
                length = episode_data["length"]

                episode_chunk = int(episode_index // chunks_size)

                parquet_path = os.path.join(
                    data_root, "data",
                    f"chunk-{episode_chunk:03d}",
                    f"episode_{episode_index:06d}.parquet",
                )
                if not os.path.exists(parquet_path):
                    zero_rank_print(f"parquet file not found: {parquet_path}")
                    continue

                # Verify required columns exist in parquet (schema-only, no data loading)
                try:
                    import pyarrow.parquet as pq
                    parquet_cols = pq.read_schema(parquet_path).names
                    required_cols = {self.action_key, self.state_key}
                    missing_cols = required_cols - set(parquet_cols)
                    if missing_cols:
                        zero_rank_print(f"parquet missing columns {missing_cols}, skipping: {parquet_path}")
                        continue
                except Exception as e:
                    zero_rank_print(f"Failed to read parquet columns for {parquet_path}: {e}")
                    continue

                video_path = os.path.join(
                    data_root, "videos",
                    f"chunk-{episode_chunk:03d}", "{}",
                    f"episode_{episode_index:06d}.mp4",
                )

                # Use Lerobot-style camera directory names
                cam_name_list = (
                    "observation.images.cam_front",
                    "observation.images.cam_left",
                    "observation.images.cam_right",
                    "observation.images.cam_down",
                )
                info = [
                    video_path,
                    cam_name_list,
                    parquet_path,
                    domain_name,
                    "",
                    caption,
                    length,
                ]
                self.dataset.append(info)

    def get_frame_indexes(self, total_frames):
        """Select n_previous memory frames + action_chunk prediction frames.

        Returns (frame_indexes, action_indexes, mem_indexes) -- mem_indexes is the
        subset of memory-frame indices, exposed separately so callers can fetch the
        pose at those exact frames for value-branch start/history conditioning.
        """
        if self.fix_sidx is not None and self.fix_mem_idx is not None:
            action_indexes = list(range(self.fix_sidx, self.fix_sidx + self.action_chunk))
            frame_indexes = action_indexes[::self.video_temporal_stride]
            action_indexes = np.clip(action_indexes, a_min=0, a_max=total_frames - 1).tolist()
            frame_indexes = np.clip(frame_indexes, a_min=0, a_max=total_frames - 1).tolist()
            return self.fix_mem_idx + frame_indexes, self.fix_mem_idx + action_indexes, self.fix_mem_idx

        if self.window_end:
            chunk_end = total_frames  # anchor so sample_n_frames window ends AT last frame; future=[N-2,N-1] distinct
        else:
            chunk_end = random.randint(self.action_chunk, total_frames + self.action_chunk)

        indexes_start = max(-self.n_previous, chunk_end - self.sample_n_frames)
        indexes = np.array(list(range(indexes_start, chunk_end)))
        indexes = np.clip(indexes, a_min=1, a_max=total_frames - 1).tolist()
        video_end = indexes[-self.action_chunk:]
        mem_candidates = indexes[:-self.action_chunk]
        if len(mem_candidates) < self.n_previous - 1:
            mem_candidates = [1] * (self.n_previous - 1) + mem_candidates

        if self.previous_pick_mode == "uniform":
            mem_indexes = [
                mem_candidates[int(i)] for i in np.linspace(
                    0, len(mem_candidates) - 1, self.n_previous
                ).tolist()
            ]
        elif self.previous_pick_mode == "random":
            mem_indexes = [
                mem_candidates[i] for i in sorted(
                    np.random.choice(
                        list(range(0, len(mem_candidates) - 1)),
                        size=self.n_previous - 1,
                        replace=False,
                    ).tolist()
                )
            ] + [mem_candidates[-1]]
        else:
            raise NotImplementedError(
                f"unsupported previous_pick_mode: {self.previous_pick_mode}"
            )

        if not self.ignore_seek:
            frame_indexes = mem_indexes + video_end[self.video_temporal_stride - 1::self.video_temporal_stride]
        else:
            frame_indexes = mem_indexes + mem_indexes[-1:]

        action_indexes = mem_indexes + video_end
        return frame_indexes, action_indexes, mem_indexes

    def get_action_bias_std(self, domain_name):
        """Get (mean, std) for normalization, extended to 14-dim."""
        mean = torch.tensor(self.StatisticInfo[domain_name + "_" + self.action_space]["mean"]).unsqueeze(0)
        std = torch.tensor(self.StatisticInfo[domain_name + "_" + self.action_space]["std"]).unsqueeze(0) + 1e-6
        return mean, std

    def seek_mp4(self, video_path, cam_name_list, slices):
        """Seek video frames from per-camera mp4 files.

        Args:
            video_path: template like '.../chunk-000/{}/episode_000000.mp4'
            cam_name_list: tuple of camera dir names (e.g. 'observation.images.cam_front')
            slices: frame indices to extract

        Returns:
            List of tensors, one per camera: [c, t, h, w]
        """
        video_list = []
        for cam_name in cam_name_list:
            full_path = video_path.format(cam_name)
            if not os.path.exists(full_path):
                # Create a tensor placeholder (must match tensor type for transform_video)
                c, t, h, w = 3, len(slices), self.sample_size[0], self.sample_size[1]
                video = torch.zeros((c, t, h, w), dtype=torch.float32)
            else:
                video_reader = VideoFileClip(full_path)
                fps = video_reader.fps
                frames = []
                for idx in slices:
                    frames.append(video_reader.get_frame(float(idx) / fps))
                video_reader.close()
                video = np.stack(frames)  # [t, h, w, c]
                video = torch.from_numpy(video).permute(3, 0, 1, 2).contiguous().float() / 255.0
            video_list.append(video)
        return video_list

    def transform_video(self, videos):
        """Crop (optional) and resize videos, and normalize."""
        v = len(videos)
        new_videos = []
        for iv in range(v):
            video = videos[iv]  # [c, t, h, w]
            c, t, h, w = video.shape
            if self.random_crop:
                # Placeholder: simple resize without crop for now
                pass
            video = self.pixel_transforms_resize(video)
            new_videos.append(video)
        new_videos = torch.stack(new_videos, dim=1)  # [c, v, t, h, w]
        return new_videos

    def normalize_video(self, video):
        """Normalize video to [-1, 1].

        Input: [c, v, t, h, w]
        """
        c, v, t, h, w = video.shape
        video = self.pixel_transforms_norm(
            video.permute(1, 2, 0, 3, 4).reshape(-1, c, h, w)
        ).reshape(v, t, c, h, w).permute(2, 0, 1, 3, 4)
        return video

    def extend_to_14(self, x):
        """Extend a 7-dim vector to 14-dim by zero-padding."""
        if x.shape[-1] == 14:
            return x
        if x.shape[-1] == 7:
            return torch.cat([x, torch.zeros_like(x[..., :7])], dim=-1)
        # Generic: pad or truncate to 14
        if x.shape[-1] < 14:
            pad = torch.zeros((*x.shape[:-1], 14 - x.shape[-1]), dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=-1)
        return x[..., :14]

    def get_batch(self, idx):
        video_path = self.dataset[idx][0]
        cam_name_list = self.dataset[idx][1]
        parquet_path = self.dataset[idx][2]
        domain_name = self.dataset[idx][3]
        caption = self.dataset[idx][5]
        total_frames = self.dataset[idx][6]

        vid_indexes, indexes, mem_indexes = self.get_frame_indexes(total_frames)
        try:
            _bp = os.path.basename(str(parquet_path))
            self._last_episode_index = int(_bp.split("_")[1].split(".")[0])
        except Exception:
            self._last_episode_index = idx
        self._last_future_idxs = vid_indexes[-self.chunk:]

        # Real (unwindowed) episode frame ids for every video-branch token sampled
        # (mem frames + future/chunk frames), so the video rope can encode true
        # elapsed time between sparsely/non-uniformly sampled frames instead of
        # just sequential token order.
        frame_ids = torch.FloatTensor(vid_indexes)
        if self._add_start_video:
            # Prepend frame_id=0 for the start frame so video-branch rope
            # encodes its true temporal position (episode origin).
            frame_ids = torch.cat([torch.FloatTensor([0.0]), frame_ids])

        # Load parquet
        data = pd.read_parquet(parquet_path)

        # Extract action and state arrays
        action = np.stack([data[self.action_key][i] for i in range(len(data))])
        state = np.stack([data[self.state_key][i] for i in range(len(data))])

        action = action.astype(np.float32)
        state = state.astype(np.float32)

        # Get normalization stats (14-dim)
        action_mean, action_std = self.get_action_bias_std(domain_name)
        state_mean, state_std = self.get_action_bias_std(domain_name + "_state")

        action_mean = torch.FloatTensor(action_mean)
        action_std = torch.FloatTensor(action_std)
        state_mean = torch.FloatTensor(state_mean)
        state_std = torch.FloatTensor(state_std)

        # Select frames by indexes
        action_selected = torch.FloatTensor(action[indexes])  # [T, 7]
        state_selected = torch.FloatTensor(state[indexes])    # [T, 7]
        state_mem = state_selected[self.n_previous - 1:self.n_previous]  # [1, 7]

        # Value-branch conditioning: start pose + mem-window pose history, with their
        # real (unwindowed) frame ids so downstream can encode true elapsed time,
        # not just token order.
        pose_frame_ids = [0] + mem_indexes                                  # [1+n_previous]
        pose_history_raw = torch.FloatTensor(state[pose_frame_ids])          # [1+n_previous, 7]
        pose_dt = torch.FloatTensor(pose_frame_ids) / max(total_frames, 1)   # [1+n_previous]

        # Normalize (before extending to 14-dim, use 7-dim stats)
        state_mean7 = state_mean[..., :7]
        state_std7 = state_std[..., :7]
        action_mean7 = action_mean[..., :7]
        action_std7 = action_std[..., :7]
        state_mem = (state_mem - state_mean7) / state_std7
        pose_history_raw = (pose_history_raw - state_mean7) / state_std7

        if self.action_type == "absolute":
            action_selected = (action_selected - action_mean7) / action_std7
        elif self.action_type == "delta":
            action_curr = torch.FloatTensor(action[indexes].astype(np.float32))
            action_last = torch.FloatTensor(action[[_ - 1 for _ in indexes]].astype(np.float32))
            action_selected = action_curr - action_last
            action_selected = (action_selected - action_mean7) / action_std7
        elif self.action_type == "relative":
            action_selected = (action_selected - action_mean7) / action_std7
            action_selected = action_selected - state_mem
        else:
            raise NotImplementedError

        # Load video frames
        videos = self.seek_mp4(video_path, cam_name_list, vid_indexes)
        videos = self.transform_video(videos)
        videos = self.normalize_video(videos)

        # Load episode frame-0 image as visual anchor for value conditioning
        if self._add_start_video:
            start_videos = self.seek_mp4(video_path, cam_name_list, [0])
            start_videos = self.transform_video(start_videos)   # [c, v, 1, h, w]
            start_videos = self.normalize_video(start_videos)   # [c, v, 1, h, w], range [-1,1]
        else:
            start_videos = None

        # Extend to 14-dim (after normalization)
        action_selected = self.extend_to_14(action_selected)   # [T, 14]
        state_mem = self.extend_to_14(state_mem)               # [1, 14]
        pose_history_raw = self.extend_to_14(pose_history_raw)  # [1+n_previous, 14]
        pose_history = torch.cat([pose_history_raw, pose_dt.unsqueeze(-1)], dim=-1)  # [1+n_previous, 15]

        # Compute discrete action classes from raw (pre-normalization) values.
        raw_action = torch.FloatTensor(action[indexes])        # [T, 7]
        if self.action_discrete_mode == "categorical":
            class_values = torch.tensor(DRONE_ACTION_CATEGORICAL_VALUES, dtype=raw_action.dtype)
            dist = (raw_action[:, None, :class_values.shape[-1]] - class_values[None]).abs().sum(dim=-1)
            action_classes = dist.argmin(dim=-1, keepdim=True)
        else:
            action_classes = torch.zeros(raw_action.shape[0], DRONE_NUM_ACTIVE_DIMS, dtype=torch.long)
            for i, dim in enumerate(DRONE_ACTIVE_DIMS):
                vals = DRONE_ACTION_DISCRETE_MAP[dim]
                if len(vals) == 1:
                    action_classes[:, i] = 0
                else:
                    dist_0 = (raw_action[:, dim] - vals[0]).abs()
                    dist_1 = (raw_action[:, dim] - vals[1]).abs()
                    action_classes[:, i] = (dist_1 < dist_0).long()

        # Load value targets
        if self.value_key in data:
            state_value = np.stack([data[self.value_key][i] for i in range(len(data))])
            state_value = state_value.astype(np.float32)
        else:
            state_value = np.zeros((len(data), 1), dtype=np.float32)

        if self.value_dense:
            value = torch.FloatTensor(state_value[indexes].astype(np.float32))
        else:
            value = torch.FloatTensor(state_value[vid_indexes].astype(np.float32))

        if self.grounding_key is not None and self.grounding_key in data:
            grounding_target = np.stack([data[self.grounding_key][i] for i in range(len(data))]).astype(np.float32)
        else:
            grounding_target = np.zeros((len(data), int(self.grounding_dim)), dtype=np.float32)

        if self.grounding_dense:
            grounding = torch.FloatTensor(grounding_target[indexes].astype(np.float32))
        else:
            grounding = torch.FloatTensor(grounding_target[vid_indexes].astype(np.float32))

        if self.grounding_weight_key is not None and self.grounding_weight_key in data:
            grounding_weight_target = np.stack([data[self.grounding_weight_key][i] for i in range(len(data))]).astype(np.float32)
        else:
            grounding_weight_target = np.ones((len(data), 1), dtype=np.float32)

        if self.grounding_weight_dense:
            grounding_weight = torch.FloatTensor(grounding_weight_target[indexes].astype(np.float32))
        else:
            grounding_weight = torch.FloatTensor(grounding_weight_target[vid_indexes].astype(np.float32))

        return videos, action_selected, caption, state_mem, value, grounding, grounding_weight, action_classes, pose_history, frame_ids, start_videos

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.fix_epiidx is not None:
            video, actions, caption, state, value, grounding, grounding_weight, action_classes, pose_history, frame_ids, start_videos = self.get_batch(self.fix_epiidx)
        else:
            while True:
                try:
                    video, actions, caption, state, value, grounding, grounding_weight, action_classes, pose_history, frame_ids, start_videos = self.get_batch(idx)
                    break
                except Exception:
                    traceback.print_exc()
                    idx = random.randint(0, self.length - 1)

        sample = {
            "video": video,
            "actions": actions,
            "caption": caption,
            "state": state,
            "value_targets": value,
            "grounding_targets": grounding,
            "grounding_weights": grounding_weight,
            "action_classes": action_classes,
            "pose_history": pose_history,
            "frame_ids": frame_ids,
        }

        if start_videos is not None:
            sample["start_video"] = start_videos

        if getattr(self, "goal_label_dir", None):
            _ep = self._last_episode_index
            _h = self._label_cache.get(_ep)
            if _h is not None:
                self._label_cache.move_to_end(_ep)
            if _h is None:
                _p = os.path.join(self.goal_label_dir, f"labels_ep{_ep:06d}.npz")
                if os.path.exists(_p):
                    with np.load(_p) as _d:
                        _h = (_d["heatmaps"].astype(np.float32),
                              _d["visible"].astype(np.float32),
                              _d["conf"].astype(np.float32))
                    if self.goal_label_cache_size > 0:
                        self._label_cache[_ep] = _h
                        while len(self._label_cache) > self.goal_label_cache_size:
                            self._label_cache.popitem(last=False)
            _fut = np.array(self._last_future_idxs, dtype=np.int64)
            if _h is not None:
                hp, vi, cf = _h
                _hs = hp[:, _fut, :, :].astype(np.float32)
                # peak-normalize the heatmap; confidence scales it separately
                _hs = _hs / np.maximum(np.amax(_hs, axis=(2, 3), keepdims=True), 1e-6)
                sample["goal_heatmaps"] = torch.FloatTensor(_hs)                  # (4, chunk, 32, 32)
                sample["goal_vis"] = torch.FloatTensor(vi[:, _fut])               # (4, chunk)
                sample["goal_conf"] = torch.FloatTensor(cf[:, _fut])              # (4, chunk)
            else:
                _nc = len(self.valid_cam)
                sample["goal_heatmaps"] = torch.zeros((_nc, self.chunk, 32, 32), dtype=torch.float32)
                sample["goal_vis"] = torch.zeros((_nc, self.chunk), dtype=torch.float32)
                sample["goal_conf"] = torch.zeros((_nc, self.chunk), dtype=torch.float32)

        return sample
