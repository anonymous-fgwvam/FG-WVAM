#!/usr/bin/env python3
"""Annotate LeRobot drone episodes with target-image grounding features.

The teacher is used only offline. The trained FG-WVAM grounding branch then learns
to predict target-conditioned features from latent futures, so inference does
not need the teacher.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CAMERAS = (
    "observation.images.cam_front",
    "observation.images.cam_left",
    "observation.images.cam_right",
    "observation.images.cam_down",
)


def torch_dtype_for(device: str, dtype: str) -> torch.dtype:
    if device.startswith("cuda") and dtype == "bf16":
        return torch.bfloat16
    if device.startswith("cuda") and dtype == "fp16":
        return torch.float16
    return torch.float32


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_task_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^navigate\s+to\s+(the|a|an)\s+", "", text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def build_object_prompt(annotation: dict, prompt_mode: str) -> str:
    category = str(annotation.get("category", "") or annotation.get("true_name", "")).strip()
    description = str(annotation.get("description", "")).strip()
    landmark = str(annotation.get("Landmark", "")).strip()

    if prompt_mode == "category":
        raw_parts = [category]
    elif prompt_mode == "description":
        raw_parts = [description or category]
    elif prompt_mode == "category_landmark":
        raw_parts = [category, landmark]
    elif prompt_mode == "category_description":
        raw_parts = [category, description]
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    parts = []
    for value in raw_parts:
        if value and value.lower() not in {part.lower() for part in parts}:
            parts.append(value)
    prompt = ". ".join(parts).lower()
    return prompt if prompt.endswith(".") else f"{prompt}."


def load_annotation_map(annotation_json: Path, goal_image_dir: Path, prompt_mode: str) -> Tuple[Dict[str, dict], Dict[int, Path], Dict[int, str]]:
    annotations = json.loads(annotation_json.read_text(encoding="utf-8"))
    desc_to_ann = {}
    object_to_image = {}
    object_to_prompt = {}
    for ann in annotations:
        object_id = int(ann["object_id"])
        desc_to_ann[normalize_task_text(ann["description"])] = ann
        object_to_prompt[object_id] = build_object_prompt(ann, prompt_mode)

        candidates = [
            goal_image_dir / f"object_{object_id}.png",
            annotation_json.parent / str(ann.get("Image", "")),
            annotation_json.parent / "images" / f"object_{object_id}.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                object_to_image[object_id] = candidate
                break
    return desc_to_ann, object_to_image, object_to_prompt


def build_task_object_map(data_root: Path, annotation_json: Path, goal_image_dir: Path, prompt_mode: str) -> Tuple[Dict[int, int], Dict[int, Path], Dict[int, str]]:
    desc_to_ann, object_to_image, object_to_prompt = load_annotation_map(annotation_json, goal_image_dir, prompt_mode)
    desc_keys = list(desc_to_ann.keys())
    task_object = {}
    tasks_path = data_root / "meta" / "tasks.jsonl"
    for task in load_jsonl(tasks_path):
        task_index = int(task["task_index"])
        key = normalize_task_text(task["task"])
        ann = desc_to_ann.get(key)
        if ann is None:
            matches = difflib.get_close_matches(key, desc_keys, n=1, cutoff=0.82)
            if matches:
                ann = desc_to_ann[matches[0]]
        if ann is None:
            raise ValueError(f"Could not map task_index={task_index} to an object id: {task['task']}")
        object_id = int(ann["object_id"])
        if object_id not in object_to_image:
            raise FileNotFoundError(f"No goal image found for object_{object_id}")
        task_object[task_index] = object_id
    return task_object, object_to_image, object_to_prompt


class ImageTeacher:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        model_family: str,
        device: str,
        dtype: str,
        local_files_only: bool,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

        source = model_path or model_name
        family = model_family
        if family == "auto":
            lower = source.lower()
            family = "clip" if "clip" in lower else "generic"

        torch_dtype = torch_dtype_for(device, dtype)

        self.family = family
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.model_name = source

        if family == "clip":
            self.processor = CLIPProcessor.from_pretrained(source, local_files_only=local_files_only)
            self.model = CLIPModel.from_pretrained(
                source,
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
            )
        else:
            try:
                self.processor = AutoImageProcessor.from_pretrained(source, local_files_only=local_files_only)
            except Exception:
                self.processor = AutoProcessor.from_pretrained(source, local_files_only=local_files_only)
            self.model = AutoModel.from_pretrained(
                source,
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
            )

        self.model.eval().to(self.device)

    @torch.inference_mode()
    def embed(self, images: Sequence[Image.Image], batch_size: int) -> torch.Tensor:
        features = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.torch_dtype)
            if hasattr(self.model, "get_image_features"):
                feat = self.model.get_image_features(pixel_values=pixel_values)
            else:
                outputs = self.model(pixel_values=pixel_values)
                feat = getattr(outputs, "pooler_output", None)
                if feat is None:
                    feat = outputs.last_hidden_state[:, 0]
            feat = F.normalize(feat.float(), dim=-1)
            features.append(feat.cpu())
        return torch.cat(features, dim=0)


def center_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    half_w = width / 2
    half_h = height / 2
    return torch.stack((center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h), dim=-1)


def clamp_box(box: Sequence[float], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    width, height = image_size
    x0 = int(round(min(max(float(box[0]), 0.0), float(width - 1))))
    y0 = int(round(min(max(float(box[1]), 0.0), float(height - 1))))
    x1 = int(round(min(max(float(box[2]), 1.0), float(width))))
    y1 = int(round(min(max(float(box[3]), 1.0), float(height))))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return x0, y0, x1, y1


class Owlv2ImageGuidedTeacher:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str,
        dtype: str,
        local_files_only: bool,
        feature_pooling: str,
        softmax_temperature: float,
    ) -> None:
        from transformers import AutoProcessor, Owlv2ForObjectDetection

        source = model_path or model_name
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype_for(device, dtype)
        self.processor = AutoProcessor.from_pretrained(source, local_files_only=local_files_only)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            source,
            torch_dtype=self.torch_dtype,
            local_files_only=local_files_only,
        )
        self.model.eval().to(self.device)
        self.feature_pooling = feature_pooling
        self.softmax_temperature = max(float(softmax_temperature), 1e-6)

    @torch.inference_mode()
    def score_frames(
        self,
        images: Sequence[Image.Image],
        query_image: Image.Image,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
        all_scores = []
        all_features = []
        all_boxes = []
        for start in range(0, len(images), batch_size):
            batch = [image.convert("RGB") for image in images[start : start + batch_size]]
            queries = [query_image.convert("RGB") for _ in batch]
            inputs = self.processor(images=batch, query_images=queries, return_tensors="pt")
            inputs = {
                key: value.to(device=self.device, dtype=self.torch_dtype) if torch.is_floating_point(value) else value.to(self.device)
                for key, value in inputs.items()
            }
            outputs = self.model.image_guided_detection(**inputs, return_dict=True)

            logits = outputs.logits.float()
            patch_logits = logits.max(dim=-1).values if logits.ndim == 3 else logits
            patch_scores = torch.sigmoid(patch_logits)
            best_scores, best_indices = patch_scores.max(dim=1)

            class_embeds = F.normalize(outputs.class_embeds.float(), dim=-1)
            if self.feature_pooling == "softmax":
                weights = torch.softmax(patch_logits / self.softmax_temperature, dim=1)
                frame_features = torch.sum(weights.unsqueeze(-1) * class_embeds, dim=1)
                frame_features = F.normalize(frame_features, dim=-1)
            elif self.feature_pooling == "top":
                frame_features = class_embeds[torch.arange(class_embeds.shape[0], device=self.device), best_indices]
            else:
                raise ValueError(f"Unknown owl_v2 feature pooling: {self.feature_pooling}")

            boxes = center_to_corners(outputs.target_pred_boxes.float())
            batch_indices = torch.arange(boxes.shape[0], device=self.device)
            best_boxes = boxes[batch_indices, best_indices].detach().cpu().numpy()
            for image, box in zip(batch, best_boxes):
                width, height = image.size
                scaled = (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
                all_boxes.append(clamp_box(scaled, image.size))

            all_scores.append(best_scores.detach().cpu().numpy().astype(np.float32))
            all_features.append(frame_features.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(all_scores, axis=0), np.concatenate(all_features, axis=0), all_boxes


class GroundingDinoTextTeacher:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str,
        dtype: str,
        local_files_only: bool,
    ) -> None:
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        source = model_path or model_name
        self.device = torch.device(device)
        if dtype != "fp32":
            print("GroundingDINO teacher uses fp32 to avoid mixed dtype issues in the text-vision encoder.")
        self.torch_dtype = torch.float32
        self.processor = AutoProcessor.from_pretrained(source, local_files_only=local_files_only)
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            source,
            torch_dtype=self.torch_dtype,
            local_files_only=local_files_only,
        )
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def score_frames(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
        all_scores = []
        all_features = []
        all_boxes = []
        for start in range(0, len(images), batch_size):
            batch = [image.convert("RGB") for image in images[start : start + batch_size]]
            texts = [prompt for _ in batch]
            inputs = self.processor(images=batch, text=texts, return_tensors="pt")
            inputs = {
                key: value.to(device=self.device, dtype=self.torch_dtype) if torch.is_floating_point(value) else value.to(self.device)
                for key, value in inputs.items()
            }
            outputs = self.model(**inputs, return_dict=True)

            query_scores = torch.sigmoid(outputs.logits.float()).max(dim=-1).values
            best_scores, best_indices = query_scores.max(dim=1)
            batch_indices = torch.arange(query_scores.shape[0], device=self.device)

            frame_features = outputs.last_hidden_state.float()[batch_indices, best_indices]
            frame_features = F.normalize(frame_features, dim=-1)

            boxes = center_to_corners(outputs.pred_boxes.float())
            best_boxes = boxes[batch_indices, best_indices].detach().cpu().numpy()
            for image, box in zip(batch, best_boxes):
                width, height = image.size
                scaled = (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
                all_boxes.append(clamp_box(scaled, image.size))

            all_scores.append(best_scores.detach().cpu().numpy().astype(np.float32))
            all_features.append(frame_features.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(all_scores, axis=0), np.concatenate(all_features, axis=0), all_boxes


def image_to_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def make_frame_crops(image: Image.Image, crop_mode: str) -> List[Tuple[Image.Image, str, Tuple[int, int, int, int]]]:
    full_box = (0, 0, image.size[0], image.size[1])
    if crop_mode == "full":
        return [(image, "full", full_box)]

    width, height = image.size
    crops = [(image, "full", full_box)]

    def crop_at(cx: float, cy: float, scale: float, name: str) -> Tuple[Image.Image, str, Tuple[int, int, int, int]]:
        cw = max(8, int(width * scale))
        ch = max(8, int(height * scale))
        left = min(max(0, int(cx * width - cw / 2)), width - cw)
        top = min(max(0, int(cy * height - ch / 2)), height - ch)
        box = (left, top, left + cw, top + ch)
        return image.crop(box), name, box

    if crop_mode == "five":
        for name, cx, cy in (("center", 0.5, 0.5), ("left", 0.25, 0.5), ("right", 0.75, 0.5), ("top", 0.5, 0.25), ("bottom", 0.5, 0.75)):
            crops.append(crop_at(cx, cy, 0.65, name))
    elif crop_mode == "nine":
        for cy in (0.25, 0.5, 0.75):
            for cx in (0.25, 0.5, 0.75):
                crops.append(crop_at(cx, cy, 0.55, f"{cx:.2f}_{cy:.2f}"))
    else:
        raise ValueError(f"Unknown crop_mode: {crop_mode}")
    return crops


def iter_video_frames(video_path: Path, expected_len: int) -> Iterable[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    last = None
    count = 0
    while count < expected_len:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last = Image.fromarray(frame)
        yield last
        count += 1

    cap.release()

    if last is not None:
        while count < expected_len:
            yield last.copy()
            count += 1


def score_episode(
    teacher: ImageTeacher,
    goal_feature: torch.Tensor,
    data_root: Path,
    episode_index: int,
    chunks_size: int,
    n_frames: int,
    cameras: Sequence[str],
    crop_mode: str,
    batch_size: int,
    view_fusion: str,
    view_fusion_temperature: float,
    collect_debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    episode_chunk = episode_index // chunks_size
    frame_candidates: List[List[dict]] = [[] for _ in range(n_frames)]
    debug_records = []
    debug_images = [None for _ in range(n_frames)] if collect_debug else None
    debug_full_images = [None for _ in range(n_frames)] if collect_debug else None
    debug_meta = [None for _ in range(n_frames)] if collect_debug else None

    for camera in cameras:
        video_path = (
            data_root
            / "videos"
            / f"chunk-{episode_chunk:03d}"
            / camera
            / f"episode_{episode_index:06d}.mp4"
        )
        if not video_path.exists():
            continue

        camera_scores = np.full((n_frames,), -1.0, dtype=np.float32)
        camera_features = None
        camera_meta = [None for _ in range(n_frames)]
        camera_images = [None for _ in range(n_frames)] if collect_debug else None
        pending_images: List[Image.Image] = []
        pending_full_images: List[Image.Image] = []
        pending_frame_ids: List[int] = []
        pending_meta: List[Tuple[str, str, Tuple[int, int, int, int]]] = []

        def flush_camera() -> None:
            nonlocal pending_images, pending_full_images, pending_frame_ids, pending_meta, camera_scores, camera_features
            if not pending_images:
                return
            features = teacher.embed(pending_images, batch_size=batch_size)
            if camera_features is None:
                camera_features = np.zeros((n_frames, features.shape[-1]), dtype=np.float32)
            sims = (features @ goal_feature.T).squeeze(-1).numpy().astype(np.float32)
            for row_idx, (frame_id, score) in enumerate(zip(pending_frame_ids, sims)):
                if score > camera_scores[frame_id]:
                    camera_scores[frame_id] = score
                    camera_features[frame_id] = features[row_idx].numpy().astype(np.float32)
                    camera_meta[frame_id] = pending_meta[row_idx]
                    if collect_debug:
                        camera_images[frame_id] = pending_full_images[row_idx].copy()
            pending_images = []
            pending_full_images = []
            pending_frame_ids = []
            pending_meta = []

        for frame_id, image in enumerate(iter_video_frames(video_path, n_frames)):
            for crop, crop_name, crop_box in make_frame_crops(image, crop_mode):
                pending_images.append(crop)
                if collect_debug:
                    pending_full_images.append(image)
                pending_frame_ids.append(frame_id)
                pending_meta.append((camera, crop_name, crop_box))
            if len(pending_images) >= batch_size:
                flush_camera()
        flush_camera()

        if camera_features is None:
            continue
        for frame_id in range(n_frames):
            if camera_scores[frame_id] <= -0.99 or camera_meta[frame_id] is None:
                continue
            _, crop_name, crop_box = camera_meta[frame_id]
            frame_candidates[frame_id].append(
                {
                    "score": float(camera_scores[frame_id]),
                    "feature": camera_features[frame_id],
                    "crop_box": crop_box,
                    "camera": camera,
                    "crop_name": crop_name,
                    "image": camera_images[frame_id] if collect_debug else None,
                }
            )

    if not any(frame_candidates):
        raise FileNotFoundError(f"No usable videos for episode {episode_index} in {data_root}")

    feature_dim = next(candidate["feature"].shape[-1] for candidates in frame_candidates for candidate in candidates)
    output_dim = feature_dim * len(cameras) if view_fusion == "concat" else feature_dim
    frame_scores = np.full((n_frames,), -1.0, dtype=np.float32)
    frame_features = np.zeros((n_frames, output_dim), dtype=np.float32)
    for frame_id, candidates in enumerate(frame_candidates):
        if not candidates:
            continue
        score, feature, record = fuse_grounding_candidates(candidates, view_fusion, view_fusion_temperature)
        frame_scores[frame_id] = score
        frame_features[frame_id] = feature
        if collect_debug:
            crop_box = record["crop_box"]
            image = record["image"]
            debug_images[frame_id] = image.crop(crop_box).copy()
            debug_full_images[frame_id] = image.copy()
            debug_meta[frame_id] = record

    if collect_debug:
        for frame_id, score in enumerate(frame_scores.tolist()):
            record = debug_meta[frame_id]
            debug_records.append(
                {
                    "frame_index": frame_id,
                    "score": float(score),
                    "camera": record["camera"],
                    "crop_name": record["crop_name"],
                    "crop_box": list(record["crop_box"]),
                    "view_fusion": record["view_fusion"],
                    "view_weight": record["view_weight"],
                    "view_candidates": record["view_candidates"],
                }
            )
    return frame_scores, frame_features, {"records": debug_records, "images": debug_images, "full_images": debug_full_images}


def fuse_grounding_candidates(
    candidates: Sequence[dict],
    view_fusion: str,
    temperature: float,
) -> Tuple[float, np.ndarray, dict]:
    if not candidates:
        raise ValueError("Cannot fuse empty grounding candidates")

    scores = np.asarray([float(candidate["score"]) for candidate in candidates], dtype=np.float32)
    best_idx = int(np.argmax(scores))
    if view_fusion == "concat":
        weights = np.ones_like(scores, dtype=np.float32)
        fused_feature = np.concatenate([candidate["feature"].astype(np.float32) for candidate in candidates], axis=0)
        norm = float(np.linalg.norm(fused_feature))
        if norm > 1e-6:
            fused_feature /= norm
    elif view_fusion == "max" or len(candidates) == 1:
        weights = np.zeros_like(scores, dtype=np.float32)
        weights[best_idx] = 1.0
        fused_feature = candidates[best_idx]["feature"].astype(np.float32)
    elif view_fusion == "softmax":
        temp = max(float(temperature), 1e-6)
        logits = (scores - float(scores.max())) / temp
        weights = np.exp(logits).astype(np.float32)
        weights /= max(float(weights.sum()), 1e-6)
        features = np.stack([candidate["feature"] for candidate in candidates], axis=0).astype(np.float32)
        fused_feature = np.sum(weights[:, None] * features, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(fused_feature))
        if norm > 1e-6:
            fused_feature /= norm
    else:
        raise ValueError(f"Unknown view_fusion: {view_fusion}")

    best = dict(candidates[best_idx])
    best["score"] = float(scores[best_idx])
    best["view_fusion"] = view_fusion
    best["view_weight"] = float(weights[best_idx])
    best["view_candidates"] = [
        {
            "camera": str(candidate["camera"]),
            "crop_name": str(candidate["crop_name"]),
            "score": float(score),
            "weight": float(weight),
            "crop_box": [int(x) for x in candidate["crop_box"]],
        }
        for candidate, score, weight in zip(candidates, scores.tolist(), weights.tolist())
    ]
    return float(scores[best_idx]), fused_feature, best


def score_episode_owlv2(
    teacher: Owlv2ImageGuidedTeacher,
    goal_image: Image.Image,
    data_root: Path,
    episode_index: int,
    chunks_size: int,
    n_frames: int,
    cameras: Sequence[str],
    batch_size: int,
    view_fusion: str,
    view_fusion_temperature: float,
    collect_debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    episode_chunk = episode_index // chunks_size
    frame_candidates: List[List[dict]] = [[] for _ in range(n_frames)]
    debug_records = []
    debug_images = [None for _ in range(n_frames)] if collect_debug else None
    debug_full_images = [None for _ in range(n_frames)] if collect_debug else None
    debug_meta = [None for _ in range(n_frames)] if collect_debug else None

    for camera in cameras:
        video_path = (
            data_root
            / "videos"
            / f"chunk-{episode_chunk:03d}"
            / camera
            / f"episode_{episode_index:06d}.mp4"
        )
        if not video_path.exists():
            continue

        frames = list(iter_video_frames(video_path, n_frames))
        if not frames:
            continue
        scores, features, boxes = teacher.score_frames(frames, goal_image, batch_size=batch_size)

        for frame_id, score in enumerate(scores.tolist()):
            crop_box = boxes[frame_id]
            frame_candidates[frame_id].append(
                {
                    "score": float(score),
                    "feature": features[frame_id],
                    "crop_box": crop_box,
                    "camera": camera,
                    "crop_name": "owlv2_box",
                    "image": frames[frame_id] if collect_debug else None,
                }
            )

    if not any(frame_candidates):
        raise FileNotFoundError(f"No usable videos for episode {episode_index} in {data_root}")

    feature_dim = next(candidate["feature"].shape[-1] for candidates in frame_candidates for candidate in candidates)
    frame_scores = np.full((n_frames,), -1.0, dtype=np.float32)
    frame_features = np.zeros((n_frames, feature_dim), dtype=np.float32)
    for frame_id, candidates in enumerate(frame_candidates):
        if not candidates:
            continue
        score, feature, record = fuse_grounding_candidates(candidates, view_fusion, view_fusion_temperature)
        frame_scores[frame_id] = score
        frame_features[frame_id] = feature
        if collect_debug:
            crop_box = record["crop_box"]
            image = record["image"]
            debug_images[frame_id] = image.crop(crop_box).copy()
            debug_full_images[frame_id] = image.copy()
            debug_meta[frame_id] = record

    if collect_debug:
        for frame_id, score in enumerate(frame_scores.tolist()):
            record = debug_meta[frame_id]
            debug_records.append(
                {
                    "frame_index": frame_id,
                    "score": float(score),
                    "camera": record["camera"],
                    "crop_name": record["crop_name"],
                    "crop_box": list(record["crop_box"]),
                    "view_fusion": record["view_fusion"],
                    "view_weight": record["view_weight"],
                    "view_candidates": record["view_candidates"],
                }
            )
    return frame_scores, frame_features, {"records": debug_records, "images": debug_images, "full_images": debug_full_images}


def score_episode_grounding_dino(
    teacher: GroundingDinoTextTeacher,
    prompt: str,
    data_root: Path,
    episode_index: int,
    chunks_size: int,
    n_frames: int,
    cameras: Sequence[str],
    batch_size: int,
    view_fusion: str,
    view_fusion_temperature: float,
    collect_debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    episode_chunk = episode_index // chunks_size
    frame_candidates: List[List[dict]] = [[] for _ in range(n_frames)]
    debug_records = []
    debug_images = [None for _ in range(n_frames)] if collect_debug else None
    debug_full_images = [None for _ in range(n_frames)] if collect_debug else None
    debug_meta = [None for _ in range(n_frames)] if collect_debug else None

    for camera in cameras:
        video_path = (
            data_root
            / "videos"
            / f"chunk-{episode_chunk:03d}"
            / camera
            / f"episode_{episode_index:06d}.mp4"
        )
        if not video_path.exists():
            continue

        frames = list(iter_video_frames(video_path, n_frames))
        if not frames:
            continue
        scores, features, boxes = teacher.score_frames(frames, prompt, batch_size=batch_size)

        for frame_id, score in enumerate(scores.tolist()):
            crop_box = boxes[frame_id]
            frame_candidates[frame_id].append(
                {
                    "score": float(score),
                    "feature": features[frame_id],
                    "crop_box": crop_box,
                    "camera": camera,
                    "crop_name": "gdino_box",
                    "image": frames[frame_id] if collect_debug else None,
                }
            )

    if not any(frame_candidates):
        raise FileNotFoundError(f"No usable videos for episode {episode_index} in {data_root}")

    feature_dim = next(candidate["feature"].shape[-1] for candidates in frame_candidates for candidate in candidates)
    frame_scores = np.full((n_frames,), -1.0, dtype=np.float32)
    frame_features = np.zeros((n_frames, feature_dim), dtype=np.float32)
    for frame_id, candidates in enumerate(frame_candidates):
        if not candidates:
            continue
        score, feature, record = fuse_grounding_candidates(candidates, view_fusion, view_fusion_temperature)
        frame_scores[frame_id] = score
        frame_features[frame_id] = feature
        if collect_debug:
            crop_box = record["crop_box"]
            image = record["image"]
            debug_images[frame_id] = image.crop(crop_box).copy()
            debug_full_images[frame_id] = image.copy()
            debug_meta[frame_id] = record

    if collect_debug:
        for frame_id, score in enumerate(frame_scores.tolist()):
            record = debug_meta[frame_id]
            debug_records.append(
                {
                    "frame_index": frame_id,
                    "score": float(score),
                    "camera": record["camera"],
                    "crop_name": record["crop_name"],
                    "crop_box": list(record["crop_box"]),
                    "view_fusion": record["view_fusion"],
                    "view_weight": record["view_weight"],
                    "view_candidates": record["view_candidates"],
                    "prompt": prompt,
                }
            )
    return frame_scores, frame_features, {"records": debug_records, "images": debug_images, "full_images": debug_full_images}


def draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def save_debug_contact_sheet(
    debug_dir: Path,
    episode_index: int,
    task_index: int,
    object_id: int,
    goal_image_path: Path,
    scores: np.ndarray,
    debug_payload: dict,
    topk: int,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    records = debug_payload["records"]
    images = debug_payload["images"]
    full_images = debug_payload.get("full_images") or images
    top_ids = np.argsort(scores)[::-1][:topk].tolist()

    thumb = 160
    margin = 12
    label_h = 42
    cols = len(top_ids) + 1
    sheet = Image.new("RGB", (cols * thumb + (cols + 1) * margin, thumb + label_h + 2 * margin), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    goal = image_to_rgb(goal_image_path).resize((thumb, thumb))
    sheet.paste(goal, (margin, margin + label_h))
    draw_label(draw, (margin, margin), f"goal object_{object_id}")

    manifest_rows = []
    for col, frame_id in enumerate(top_ids, start=1):
        image = full_images[frame_id]
        if image is None:
            continue
        record = records[frame_id]
        crop_box = record["crop_box"]
        image = image.convert("RGB")
        canvas = Image.new("RGB", (thumb, thumb), (235, 235, 235))
        scale = min(thumb / image.size[0], thumb / image.size[1])
        new_size = (max(1, int(round(image.size[0] * scale))), max(1, int(round(image.size[1] * scale))))
        resized = image.resize(new_size)
        offset = ((thumb - new_size[0]) // 2, (thumb - new_size[1]) // 2)
        canvas.paste(resized, offset)
        box = [
            int(round(crop_box[0] * scale)) + offset[0],
            int(round(crop_box[1] * scale)) + offset[1],
            int(round(crop_box[2] * scale)) + offset[0],
            int(round(crop_box[3] * scale)) + offset[1],
        ]
        ImageDraw.Draw(canvas).rectangle(box, outline=(255, 40, 40), width=3)
        x = margin + col * (thumb + margin)
        y = margin + label_h
        sheet.paste(canvas, (x, y))
        label = f"f{frame_id} {record['score']:.3f} {record['camera'].split('.')[-1]} {record['crop_name']}"
        draw_label(draw, (x, margin), label)
        manifest_rows.append(record)

    prefix = f"episode_{episode_index:06d}_task_{task_index:03d}_object_{object_id}"
    sheet.save(debug_dir / f"{prefix}.jpg", quality=95)
    (debug_dir / f"{prefix}.json").write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")


def make_grounding_targets(scores: np.ndarray, features: np.ndarray, future_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if future_mode == "frame":
        values = scores
        future_features = features
    elif future_mode == "suffix_max":
        values = np.empty_like(scores)
        future_features = np.empty_like(features)
        best_score = -float("inf")
        best_idx = len(scores) - 1
        for idx in range(len(scores) - 1, -1, -1):
            if float(scores[idx]) >= best_score:
                best_score = float(scores[idx])
                best_idx = idx
            values[idx] = best_score
            future_features[idx] = features[best_idx]
    else:
        raise ValueError(f"Unknown future_mode: {future_mode}")
    return values.astype(np.float32), future_features.astype(np.float32)


def resolve_teacher_family(model_name: str, model_path: str, family: str) -> str:
    if family != "auto":
        return family
    source = (model_path or model_name).lower()
    if "grounding-dino" in source or "grounding_dino" in source or "groundingdino" in source:
        return "grounding_dino_text"
    if "owlv2" in source or "owl-v2" in source:
        return "owlv2_image_guided"
    if "clip" in source:
        return "clip"
    return "generic"


def prepare_output_root(data_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    meta_src = data_root / "meta"
    meta_dst = output_root / "meta"
    if meta_dst.exists():
        shutil.rmtree(meta_dst)
    shutil.copytree(meta_src, meta_dst)

    videos_dst = output_root / "videos"
    if not videos_dst.exists():
        videos_dst.symlink_to(data_root / "videos", target_is_directory=True)
    elif videos_dst.is_symlink() and videos_dst.resolve() != (data_root / "videos").resolve():
        raise RuntimeError(f"{videos_dst} points to {videos_dst.resolve()}, expected {data_root / 'videos'}")

    (output_root / "data").mkdir(exist_ok=True)


def iter_parquets(data_root: Path) -> List[Path]:
    return sorted((data_root / "data").glob("chunk-*/*.parquet"))


def annotate_dataset(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    annotation_json = Path(args.annotation_json)
    goal_image_dir = Path(args.goal_image_dir)

    task_object, object_to_image, object_to_prompt = build_task_object_map(
        data_root,
        annotation_json,
        goal_image_dir,
        args.grounding_dino_prompt_mode,
    )
    parquets = iter_parquets(data_root)
    if args.limit is not None:
        parquets = parquets[: args.limit]

    info_path = data_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    chunks_size = int(info.get("chunks_size", 1000))

    print(f"data_root={data_root}")
    print(f"output_root={output_root}")
    print(f"episodes_to_process={len(parquets)}")
    for p in parquets[:5]:
        df = pd.read_parquet(p)
        task_index = int(df["task_index"].iloc[0])
        object_id = task_object[task_index]
        print(
            f"check episode={int(df['episode_index'].iloc[0])} task={task_index} object={object_id} "
            f"goal={object_to_image[object_id]} prompt={object_to_prompt[object_id]}"
        )

    if args.dry_run:
        return

    prepare_output_root(data_root, output_root)
    teacher_family = resolve_teacher_family(args.teacher_model, args.teacher_model_path, args.teacher_family)
    if teacher_family == "owlv2_image_guided":
        teacher = Owlv2ImageGuidedTeacher(
            model_name=args.teacher_model,
            model_path=args.teacher_model_path,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            feature_pooling=args.owlv2_feature_pooling,
            softmax_temperature=args.owlv2_softmax_temperature,
        )
    elif teacher_family == "grounding_dino_text":
        teacher = GroundingDinoTextTeacher(
            model_name=args.teacher_model,
            model_path=args.teacher_model_path,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
    else:
        teacher = ImageTeacher(
            model_name=args.teacher_model,
            model_path=args.teacher_model_path,
            model_family=teacher_family,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
        )

    goal_cache: Dict[int, object] = {}

    for idx, parquet_path in enumerate(parquets):
        relative = parquet_path.relative_to(data_root / "data")
        out_path = output_root / "data" / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            continue

        df = pd.read_parquet(parquet_path)
        if df.empty:
            continue

        episode_index = int(df["episode_index"].iloc[0])
        task_index = int(df["task_index"].iloc[0])
        object_id = task_object[task_index]

        if object_id not in goal_cache:
            goal_image = image_to_rgb(object_to_image[object_id])
            if teacher_family == "owlv2_image_guided":
                goal_cache[object_id] = goal_image
            elif teacher_family == "grounding_dino_text":
                goal_cache[object_id] = object_to_prompt[object_id]
            else:
                goal_cache[object_id] = teacher.embed([goal_image], batch_size=1)

        collect_debug = bool(args.debug_dir) and (idx < args.debug_max_episodes)
        if teacher_family == "owlv2_image_guided":
            scores, features, debug_payload = score_episode_owlv2(
                teacher=teacher,
                goal_image=goal_cache[object_id],
                data_root=data_root,
                episode_index=episode_index,
                chunks_size=chunks_size,
                n_frames=len(df),
                cameras=args.cameras,
                batch_size=args.batch_size,
                view_fusion=args.view_fusion,
                view_fusion_temperature=args.view_fusion_temperature,
                collect_debug=collect_debug,
            )
        elif teacher_family == "grounding_dino_text":
            scores, features, debug_payload = score_episode_grounding_dino(
                teacher=teacher,
                prompt=goal_cache[object_id],
                data_root=data_root,
                episode_index=episode_index,
                chunks_size=chunks_size,
                n_frames=len(df),
                cameras=args.cameras,
                batch_size=args.batch_size,
                view_fusion=args.view_fusion,
                view_fusion_temperature=args.view_fusion_temperature,
                collect_debug=collect_debug,
            )
        else:
            scores, features, debug_payload = score_episode(
                teacher=teacher,
                goal_feature=goal_cache[object_id],
                data_root=data_root,
                episode_index=episode_index,
                chunks_size=chunks_size,
                n_frames=len(df),
                cameras=args.cameras,
                crop_mode=args.crop_mode,
                batch_size=args.batch_size,
                view_fusion=args.view_fusion,
                view_fusion_temperature=args.view_fusion_temperature,
                collect_debug=collect_debug,
            )
        values, feature_targets = make_grounding_targets(scores, features, args.future_mode)
        if collect_debug:
            save_debug_contact_sheet(
                debug_dir=Path(args.debug_dir),
                episode_index=episode_index,
                task_index=task_index,
                object_id=object_id,
                goal_image_path=object_to_image[object_id],
                scores=scores,
                debug_payload=debug_payload,
                topk=args.debug_topk,
            )

        df[args.score_key] = [[float(x)] for x in scores]
        df[args.value_score_key] = [[float(x)] for x in values]
        df[args.value_key] = [[float(v) for v in row] for row in feature_targets]
        df["grounding_object_id"] = object_id
        df.to_parquet(out_path, index=False)

        if (idx + 1) % args.log_every == 0 or idx == 0:
            print(
                f"[{idx + 1}/{len(parquets)}] episode={episode_index} object={object_id} "
                f"score_range=({float(scores.min()):.4f},{float(scores.max()):.4f}) "
                f"value_range=({float(values.min()):.4f},{float(values.max()):.4f})",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--annotation-json", default="/path/to/data/AerialBench/pipelines/Task_Design/Object_annotation.json")
    parser.add_argument("--goal-image-dir", default="/path/to/data/AerialBench/pipelines/Task_Design/images")
    parser.add_argument("--teacher-model", default="facebook/dinov2-base")
    parser.add_argument("--teacher-model-path", default="")
    parser.add_argument("--teacher-family", choices=("auto", "clip", "generic", "owlv2_image_guided", "grounding_dino_text"), default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--crop-mode", choices=("full", "five", "nine"), default="five")
    parser.add_argument("--view-fusion", choices=("concat", "softmax", "max"), default="concat")
    parser.add_argument("--view-fusion-temperature", type=float, default=0.1)
    parser.add_argument("--grounding-dino-prompt-mode", choices=("category", "description", "category_description", "category_landmark"), default="category")
    parser.add_argument("--owlv2-feature-pooling", choices=("softmax", "top"), default="softmax")
    parser.add_argument("--owlv2-softmax-temperature", type=float, default=1.0)
    parser.add_argument("--future-mode", choices=("frame", "suffix_max"), default="suffix_max")
    parser.add_argument("--value-key", default="grounding_feature")
    parser.add_argument("--value-score-key", default="grounding_value")
    parser.add_argument("--score-key", default="grounding_score")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-dir", default="")
    parser.add_argument("--debug-max-episodes", type=int, default=20)
    parser.add_argument("--debug-topk", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    annotate_dataset(parse_args())
