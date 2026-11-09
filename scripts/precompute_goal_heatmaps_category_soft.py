#!/usr/bin/env python3
"""Create per-frame category heatmaps from top-k GroundingDINO boxes."""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time

_THREADS = os.environ.get("THREADS", "8")
for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ.setdefault(_name, _THREADS)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate_drone_grounding_value import center_to_corners


DEFAULT_ROOT = "/path/to/data/drone_lerobot_dataset_fantasy_v1"
DEFAULT_GDINO = "/path/to/data/hf_models/IDEA-Research_grounding-dino-base"
DEFAULT_ANNOTATION = "/path/to/data/AerialBench/pipelines/Task_Design/Object_annotation.json"
CAMERAS = [
    "observation.images.cam_front",
    "observation.images.cam_left",
    "observation.images.cam_right",
    "observation.images.cam_down",
]
GRID_SIZE = 32


def normalize_description(text):
    text = text.strip().lower()
    for article in ("a ", "an ", "the "):
        if text.startswith(article):
            return text[len(article) :]
    return text


def category_prompt(task, annotations):
    task_lower = task.lower()
    matches = [
        item
        for item in annotations
        if normalize_description(item.get("description", "")) in task_lower
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one annotation match, got {len(matches)}: {task}")
    return f"{matches[0]['category'].strip().lower()}."


def matching_annotation(task, annotations):
    task_lower = task.lower()
    matches = [
        item
        for item in annotations
        if normalize_description(item.get("description", "")) in task_lower
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one annotation match, got {len(matches)}: {task}")
    return matches[0]


def visual_phrase_prompt(task, annotations):
    annotation = matching_annotation(task, annotations)
    category = annotation["category"].strip().lower()
    first_sentence = annotation["description"].split(".", 1)[0].strip().lower()
    first_sentence = re.sub(r"^(?:a|an|the|several|two|three|four)\s+", "", first_sentence)
    clause = re.split(
        r"\b(?:with|made|built|under|containing|enclosed|joined|are|is|hangs|"
        r"sit|sits|set|mounted|composed|suspended|stretched|support|supports|"
        r"without|planted|including|on|in|beneath|above)\b",
        first_sentence,
        maxsplit=1,
    )[0].strip(" ,-")
    words = clause.split()[:8]
    normalized_words = {word.strip(" ,-").removesuffix("s") for word in words}
    category_words = {word.removesuffix("s") for word in category.split()}
    if not category_words.issubset(normalized_words):
        if words:
            words[-1:] = category.split()
        else:
            words = category.split()
    return " ".join(words) + "."


def box_iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(area_a + area_b - inter, 1e-8)


def nms_topk(scores, boxes, threshold, iou_threshold, top_k):
    order = np.flatnonzero(scores >= threshold)
    order = order[np.argsort(scores[order])[::-1]]
    kept = []
    for index in order:
        box = np.clip(boxes[index], 0.0, 1.0)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if all(box_iou(box, previous[1]) <= iou_threshold for previous in kept):
            kept.append((float(scores[index]), box.astype(np.float32)))
            if len(kept) == top_k:
                break
    return kept


def soft_box_heatmap(candidates):
    coordinates = np.linspace(0.0, 1.0, GRID_SIZE, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    heatmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for score, box in candidates:
        cx = float(box[0] + box[2]) * 0.5
        cy = float(box[1] + box[3]) * 0.5
        sigma_x = max(float(box[2] - box[0]) * 0.35, 1.0 / GRID_SIZE)
        sigma_y = max(float(box[3] - box[1]) * 0.35, 1.0 / GRID_SIZE)
        gaussian = np.exp(
            -0.5
            * (((x_grid - cx) / sigma_x) ** 2 + ((y_grid - cy) / sigma_y) ** 2)
        ).astype(np.float32)
        heatmap = np.maximum(heatmap, score * gaussian)
    return heatmap


def read_frames(video_path, expected_frames):
    capture = cv2.VideoCapture(video_path)
    frames = []
    consecutive_failures = 0
    while len(frames) < expected_frames:
        ok, frame = capture.read()
        if not ok:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
            continue
        consecutive_failures = 0
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--gdino", default=DEFAULT_GDINO)
    parser.add_argument("--annotation-json", default=DEFAULT_ANNOTATION)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episode-indices", type=int, nargs="+")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument(
        "--prompt-mode",
        choices=["category", "visual_phrase"],
        default="category",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.n_shards < 1:
        parser.error("--n-shards must be at least 1")
    if not 0 <= args.shard < args.n_shards:
        parser.error("--shard must satisfy 0 <= shard < n-shards")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    processor = AutoProcessor.from_pretrained(args.gdino, local_files_only=True)
    model = GroundingDinoForObjectDetection.from_pretrained(
        args.gdino, torch_dtype=torch.float32, local_files_only=True
    ).eval().to(device)

    with open(os.path.join(args.root, "meta", "episodes.jsonl")) as file:
        episodes = [json.loads(line) for line in file]
    with open(args.annotation_json) as file:
        annotations = json.load(file)
    indices = args.episode_indices or list(range(len(episodes)))
    indices = [index for index in indices if index % args.n_shards == args.shard]
    if args.max_episodes:
        indices = indices[: args.max_episodes]
    os.makedirs(args.out, exist_ok=True)

    @torch.inference_mode()
    def detect(frame_list, prompt):
        results = [[] for _ in frame_list]
        for start in range(0, len(frame_list), args.batch):
            batch = frame_list[start : start + args.batch]
            inputs = processor(
                images=batch,
                text=[prompt] * len(batch),
                size={"shortest_edge": args.size, "longest_edge": args.size},
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs, return_dict=True)
            boxes = center_to_corners(outputs.pred_boxes.float()).cpu().numpy()
            scores = torch.sigmoid(outputs.logits.float()).max(-1).values.cpu().numpy()
            for local_index in range(len(batch)):
                results[start + local_index] = nms_topk(
                    scores[local_index],
                    boxes[local_index],
                    args.box_threshold,
                    args.nms_iou,
                    args.top_k,
                )
        return results

    rows = []
    started = time.time()
    for position, episode_index in enumerate(indices, 1):
        episode = episodes[episode_index]
        frame_count = int(episode["length"])
        if args.prompt_mode == "visual_phrase":
            prompt = visual_phrase_prompt(episode["tasks"][0], annotations)
        else:
            prompt = category_prompt(episode["tasks"][0], annotations)
        output_path = os.path.join(args.out, f"labels_ep{episode_index:06d}.npz")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"[skip] episode={episode_index:06d}", flush=True)
            continue

        heatmaps = np.zeros((4, frame_count, GRID_SIZE, GRID_SIZE), np.float32)
        confidence = np.zeros((4, frame_count), np.float32)
        visible = np.zeros((4, frame_count), np.uint8)
        boxes_out = np.full((4, frame_count, args.top_k, 4), -1.0, np.float32)
        scores_out = np.zeros((4, frame_count, args.top_k), np.float32)
        counts_out = np.zeros((4, frame_count), np.uint8)
        chunk = episode_index // 1000

        for camera_index, camera in enumerate(CAMERAS):
            video_path = os.path.join(
                args.root,
                "videos",
                f"chunk-{chunk:03d}",
                camera,
                f"episode_{episode_index:06d}.mp4",
            )
            frames = read_frames(video_path, frame_count)
            detections = detect(frames, prompt)
            for frame_index, candidates in enumerate(detections):
                if not candidates:
                    continue
                count = len(candidates)
                visible[camera_index, frame_index] = 1
                confidence[camera_index, frame_index] = candidates[0][0]
                counts_out[camera_index, frame_index] = count
                heatmaps[camera_index, frame_index] = soft_box_heatmap(candidates)
                for candidate_index, (score, box) in enumerate(candidates):
                    boxes_out[camera_index, frame_index, candidate_index] = box
                    scores_out[camera_index, frame_index, candidate_index] = score

        temporary_file = tempfile.NamedTemporaryFile(
            prefix=f".labels_ep{episode_index:06d}.",
            suffix=".npz",
            dir=args.out,
            delete=False,
        )
        temporary_path = temporary_file.name
        temporary_file.close()
        try:
            np.savez(
                temporary_path,
                heatmaps=heatmaps,
                conf=confidence,
                visible=visible,
                n_cam=np.asarray(4),
                n_frames=np.asarray(frame_count),
                prompt=np.asarray(prompt),
                prompt_mode=np.asarray(f"{args.prompt_mode}_soft_topk"),
                box_threshold=np.asarray(args.box_threshold, np.float32),
                top_k=np.asarray(args.top_k),
                nms_iou=np.asarray(args.nms_iou, np.float32),
                boxes=boxes_out,
                box_scores=scores_out,
                box_count=counts_out,
            )
            os.replace(temporary_path, output_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        positive = visible.astype(bool)
        rows.append(
            {
                "ep": episode_index,
                "nfr": frame_count,
                "pres_pct": round(100.0 * float(visible.mean()), 1),
                "mean_conf": round(float(confidence[positive].mean()), 3) if positive.any() else 0.0,
                "mean_boxes": round(float(counts_out[positive].mean()), 2) if positive.any() else 0.0,
                "prompt": prompt,
            }
        )
        print(
            f"[{position}/{len(indices)}] ep={episode_index:06d} prompt={prompt} "
            f"visible={100 * visible.mean():.1f}%",
            flush=True,
        )

    summary_name = (
        "summary.csv"
        if args.n_shards == 1
        else f"summary_shard{args.shard:02d}_of_{args.n_shards:02d}.csv"
    )
    summary_path = os.path.join(args.out, summary_name)
    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["ep", "nfr", "pres_pct", "mean_conf", "mean_boxes", "prompt"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done in {time.time() - started:.1f}s: {args.out}", flush=True)


if __name__ == "__main__":
    main()
