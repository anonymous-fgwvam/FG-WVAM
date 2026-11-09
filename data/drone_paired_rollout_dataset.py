import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from data.drone_lerobot_dataset import DroneLeRobotDataset


def _episode_index(dataset_entry):
    parquet_path = dataset_entry[2]
    return int(os.path.basename(parquet_path).split("_")[1].split(".")[0])


class PairedRolloutBatchSampler(Sampler):
    """Keep successful and failed continuations of one observation in one batch."""

    def __init__(self, num_pairs, batch_size, shuffle=True, drop_last=True):
        if batch_size % 2:
            raise ValueError("paired rollout batch_size must be even")
        self.num_pairs = int(num_pairs)
        self.pairs_per_batch = batch_size // 2
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

    def __iter__(self):
        pair_indices = list(range(self.num_pairs))
        if self.shuffle:
            random.shuffle(pair_indices)
        for offset in range(0, self.num_pairs, self.pairs_per_batch):
            group = pair_indices[offset:offset + self.pairs_per_batch]
            if self.drop_last and len(group) < self.pairs_per_batch:
                break
            batch = []
            for pair_index in group:
                batch.extend((2 * pair_index, 2 * pair_index + 1))
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_pairs // self.pairs_per_batch
        return (self.num_pairs + self.pairs_per_batch - 1) // self.pairs_per_batch


class DronePairedRolloutDataset(Dataset):
    """Counterfactual success/failure pairs with a shared initial observation.

    `data_roots[0]` contains successful expert rollouts and `data_roots[1]`
    contains failed rollouts. Matching episode IDs share the same task, initial
    pose, and initial camera observations. The sampled future starts at frame 1,
    while the memory frames are repeated frame-0 observations for both branches.
    """

    def __init__(
        self,
        data_roots,
        domains,
        split=None,
        dataset_info_cache_path=None,
        pair_future_start=1,
        success_distance_threshold=6.0,
        share_success_memory=True,
        **dataset_kwargs,
    ):
        if len(data_roots) != 2:
            raise ValueError(
                "DronePairedRolloutDataset expects [success_root, failure_root]"
            )

        n_previous = int(dataset_kwargs.get("n_previous", 2))
        if n_previous < 1:
            raise ValueError("paired rollout training requires n_previous >= 1")
        fixed_memory = [0] * n_previous

        success_cache = None
        failure_cache = None
        if dataset_info_cache_path:
            success_cache = dataset_info_cache_path + "_success.json"
            failure_cache = dataset_info_cache_path + "_failure.json"

        common = dict(dataset_kwargs)
        common.update(
            domains=domains,
            fix_sidx=int(pair_future_start),
            fix_mem_idx=fixed_memory,
        )
        self.success_dataset = DroneLeRobotDataset(
            data_roots=[data_roots[0]],
            split=split,
            dataset_info_cache_path=success_cache,
            **common,
        )
        # The failure dataset has only a train split. Pair it by episode ID with
        # whichever success split is active.
        self.failure_dataset = DroneLeRobotDataset(
            data_roots=[data_roots[1]],
            split=None,
            dataset_info_cache_path=failure_cache,
            **common,
        )

        failure_by_episode = {
            _episode_index(entry): index
            for index, entry in enumerate(self.failure_dataset.dataset)
        }
        self.pairs = []
        for success_index, entry in enumerate(self.success_dataset.dataset):
            episode_index = _episode_index(entry)
            failure_index = failure_by_episode.get(episode_index)
            if failure_index is not None:
                self.pairs.append((episode_index, success_index, failure_index))

        if not self.pairs:
            raise RuntimeError("no matching success/failure rollout pairs were found")

        self.pair_future_start = int(pair_future_start)
        self.n_previous = n_previous
        self.action_chunk = self.success_dataset.action_chunk
        self.success_distance_threshold = float(success_distance_threshold)
        self.share_success_memory = bool(share_success_memory)

    def __len__(self):
        return 2 * len(self.pairs)

    def make_batch_sampler(self, batch_size, shuffle=True, drop_last=True):
        return PairedRolloutBatchSampler(
            len(self.pairs), batch_size, shuffle=shuffle, drop_last=drop_last
        )

    def _value_targets(self, parquet_path, goal_anchor, rollout_success):
        data = pd.read_parquet(parquet_path, columns=["observation.pose"])
        poses = np.stack(data["observation.pose"].to_numpy()).astype(np.float32)
        goal_anchor = np.asarray(goal_anchor, dtype=np.float32)
        distances = np.linalg.norm(poses[:, :3] - goal_anchor[None], axis=1)
        distance_scale = max(float(distances[0]), self.success_distance_threshold)

        future_indices = list(
            range(
                self.pair_future_start,
                self.pair_future_start + self.action_chunk,
            )
        )
        selected_indices = [0] * self.n_previous + future_indices
        selected_indices = np.clip(selected_indices, 0, len(poses) - 1)

        progress = 1.0 - distances[selected_indices] / distance_scale
        progress = np.clip(progress, 0.0, 1.0)
        # Equal-weight normalized components: true goal progress and actual
        # terminal outcome. Failed rollouts can earn progress but never receive
        # the success half of the score.
        targets = 0.5 * (progress + float(rollout_success))
        return torch.from_numpy(targets.astype(np.float32)).unsqueeze(-1)

    def __getitem__(self, index):
        pair_index = index // 2
        success_role = index % 2 == 0
        episode_index, success_index, failure_index = self.pairs[pair_index]

        failure_entry = self.failure_dataset.dataset[failure_index]
        failure_table = pd.read_parquet(
            failure_entry[2], columns=["goal_anchor", "is_success"]
        )
        goal_anchor = np.asarray(failure_table["goal_anchor"].iloc[0], dtype=np.float32)

        if success_role:
            sample = self.success_dataset[success_index]
            parquet_path = self.success_dataset.dataset[success_index][2]
            rollout_success = 1.0
        else:
            sample = self.failure_dataset[failure_index]
            parquet_path = failure_entry[2]
            rollout_success = float(failure_table["is_success"].max())
            if self.share_success_memory:
                success_entry = self.success_dataset.dataset[success_index]
                shared_memory = self.success_dataset.seek_mp4(
                    success_entry[0], success_entry[1], [0] * self.n_previous
                )
                shared_memory = self.success_dataset.transform_video(shared_memory)
                shared_memory = self.success_dataset.normalize_video(shared_memory)
                sample["video"][:, :, :self.n_previous] = shared_memory

        sample["value_targets"] = self._value_targets(
            parquet_path, goal_anchor, rollout_success
        )
        sample["pair_id"] = torch.tensor(episode_index, dtype=torch.long)
        sample["pair_role"] = torch.tensor(
            1.0 if success_role else 0.0, dtype=torch.float32
        )
        # Failed actions are counterexamples for value learning, not behavior
        # cloning targets.
        sample["action_loss_mask"] = torch.tensor(
            1.0 if success_role else 0.0, dtype=torch.float32
        )
        sample["rollout_success"] = torch.tensor(
            rollout_success, dtype=torch.float32
        )
        return sample
