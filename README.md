# FG-WVAM: World-Value-Action Model with Future Target Grounding

Code for the paper **"World-Value-Action Model with Future Target Grounding for
Language-Goal Aerial Object Navigation"**.

This repository contains the training, annotation, and serving code. The
datasets (AerialDojo, UAV-ON) and the base models (LTX-Video with its T5 and
VAE, and the Grounding DINO teacher) are publicly released; Section 10 lists
where to obtain them and how to point the configs at them.

The code builds on Genie-Envisioner and LTX-Video, and uses Grounding DINO
for training labels.

---

## 1. Repository layout

```
configs/
  paper/                     configs for the four training stages
  ltx_model/drone/           the three source configs used to derive them
  drone_stats_fantasy.json   action/state normalization statistics
data/
  drone_lerobot_dataset.py   LeRobot-style loader (4 views, pose, action, value)
  drone_paired_rollout_dataset.py
models/
  ltx_models/                multi-view LTX-Video transformer
  action_patches/            Action Expert
  value_patches/             Future Value Expert
  grounding_patches/         grounding tokens
  pipeline/                  training/inference pipeline
runner/
  trainer.py                 training loop
  inferencer.py
scripts/
  precompute_goal_heatmaps_category_soft.py   Grounding DINO teacher labels
  annotate_drone_grounding_value.py          optional grounding-feature annotator
  value_annotation/                          stage-aware value labels
  merge_drone_action_value_checkpoints.py
  serve_policy.py                   HTTP policy server
  smoke_test_argmax.py                       synthetic serving test
  eval_sim.py                                closed-loop simulator evaluation
  train_stage1.sh / train_stage2.sh / train_stage2b.sh / train_stage3.sh
utils/
demo/                       qualitative rollout video and poster
main.py
requirements.txt
```

---

## 2. Environment

Python 3.10, a CUDA-capable GPU, and the packages in `requirements.txt`.

```bash
conda create -n fgwvam python=3.10 -y
conda activate fgwvam
pip install -r requirements.txt
```

External assets you must provide:

- **LTX-Video-2B** (video VAE, T5 text encoder, transformer initialization).
  Set `pretrained_model_name_or_path` in the configs.
- **Grounding DINO-base** (`IDEA-Research/grounding-dino-base` or a local copy)
  for training labels only. It is not used at inference.
- **AerialDojo or UAV-ON data** in the layout below.

---

## 3. Data layout

The loader expects a LeRobot-style dataset:

```
dataset/
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    splits.jsonl
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/<video_key>/episode_000000.mp4
```

Each parquet row contains:

| Column | Shape | Description |
|---|---|---|
| `observation.images.cam_front` | video | front view |
| `observation.images.cam_left` | video | left view |
| `observation.images.cam_right` | video | right view |
| `observation.images.cam_down` | video | downward view |
| `observation.pose` | `(7,)` | UAV pose (value annotation uses `[:3]` position) |
| `action` | `(7,)` | discrete-action prototype values (AerialDojo) |
| `task_index` | scalar | index into `meta/tasks.jsonl` |
| `goal_anchor` | `(3,)`, optional | target position; if absent, resolved from the annotation JSON |
| `is_success` | scalar, optional | 1 for expert success, 0 for failure rollouts |

AerialDojo uses 6 discrete actions and UAV-ON uses 8; adapt
`data/drone_lerobot_dataset.py` if you train on UAV-ON.

---

## 4. Grounding labels (Grounding DINO teacher)

The teacher produces the future target heatmaps and the visibility/confidence
signals used by both the video stage and the value annotation.

```bash
python scripts/precompute_goal_heatmaps_category_soft.py \
  --root /path/to/dataset \
  --out /path/to/labels \
  --gdino /path/to/grounding-dino-base \
  --annotation-json /path/to/Object_annotation.json \
  --size 512 --batch 16 \
  --box-threshold 0.3 --top-k 3 --nms-iou 0.5
```

This writes `labels/labels_epXXXXXX.npz` with:

- `heatmaps`: `(4, T, 32, 32)` soft target heatmaps per view;
- `visible`: `(4, T)` target visibility per view;
- `conf`: `(4, T)` detection confidence per view;
- `boxes`, `box_scores`, `box_count`: raw teacher boxes.

The teacher is training-only. At inference the model predicts grounding tokens
and heatmaps itself; Grounding DINO is absent.

`scripts/annotate_drone_grounding_value.py` is an optional annotator for the
DINOv2 / OWLv2 / Grounding DINO **feature** variants (it writes
`grounding_score`, `grounding_value`, `grounding_feature`). The main paper
configs only need the heatmap labels above.

---

## 5. Stage-aware value labels

The generator is in `scripts/value_annotation/`. It reads the grounding
`visible` / `conf` signals from step 4 and writes `state_value` and
`value_phase`.

For expert trajectories (no `is_success` column):

```bash
python scripts/value_annotation/batch_v2_value.py \
  --root /path/to/dataset \
  --out /path/to/dataset_value \
  --labels /path/to/labels \
  --annotation-json /path/to/Object_annotation.json \
  --default-success
```

For failure rollouts (with `is_success = 0`), drop `--default-success`; the
script scales the exploration and approach curves down for failures.

If your parquet already contains `goal_anchor`, `--annotation-json` is optional.

The output is a complete dataset: the script copies `meta/`, symlinks
`videos/`, and writes the relabeled `data/` parquet files.

The exact formulas are in
[`scripts/value_annotation/README.md`](scripts/value_annotation/README.md).
In short:

- `t_g` = first frame of two consecutive frames with the target visible in at
  least one view;
- exploration: `e_t = clip((m + rho + i - s) / 3, 0, 1)` and
  `y_t = 0.4 * e_t`, where `m` is normalized 3D motion, `rho` the yaw change
  relative to a 15° step, `i` the positive grounding-confidence gain, and `s`
  a short-horizon stationarity term; each signal is normalized to `[0, 1]`
  first and the combination is divided by the total absolute weight;
- approach: `0.4 + 0.6 * clip(1 - d_t/d_tg, 0, 1)`;
- exploration scores lie in `[0, 0.4]`, i.e. at most the approach floor `0.4`.

---

## 6. Configure the training stages

All configs live in `configs/paper/`. Edit the following fields:

| Field | What to set |
|---|---|
| `pretrained_model_name_or_path` | path to LTX-Video-2B |
| `diffusion_model.model_path` | stage initialization checkpoint |
| `data.train.data_roots`, `data.val.data_roots` | dataset path; stage 2b uses the expert and failure value datasets, stage 3 uses the expert value dataset |
| `data.*.stat_file` | `configs/drone_stats_fantasy.json` (included) |
| `data.*.goal_label_dir` | labels directory from step 4 (stages 1 and 2) |
| `data.*.dataset_info_cache_path` | `data/cache/...` (created on first run) |
| `output_dir`, `logging_dir` | where to write runs and logs |

Stage 1 and 2 use the raw dataset; stage 2b uses the value-annotated expert and
failure datasets; stage 3 uses the value-annotated expert dataset.
For AerialDojo, `action_num_classes_per_dim` is `6`. UAV-ON uses `8`.

---

## 7. Training

The four stages run in order. Each script takes an optional config path and
reads `GPUS` (default 1) and `MASTER_PORT`.

### Stage 1: video + goal heatmap head

Set `diffusion_model.model_path` to the LTX-Video checkpoint, then:

```bash
bash scripts/train_stage1.sh configs/paper/stage1_video_paper.yaml
```

### Stage 2: grounding + target-weighted video loss

Set `diffusion_model.model_path` in `stage2_grounding_paper.yaml` to the stage-1
checkpoint, then:

```bash
bash scripts/train_stage2.sh configs/paper/stage2_grounding_paper.yaml
```

### Stage 2b: value pretraining

Set `diffusion_model.model_path` in `stage2b_value_pretrain.yaml` to the stage-2
checkpoint, then:

```bash
bash scripts/train_stage2b.sh configs/paper/stage2b_value_pretrain.yaml
```

### Stage 3: action training

Set `diffusion_model.model_path` in `stage3_action_value_paper.yaml` to the
stage-2b checkpoint. To merge a separately pretrained action/value checkpoint
first:

```bash
python scripts/merge_drone_action_value_checkpoints.py \
  --action_checkpoint /path/to/stage2b_ckpt \
  --value_checkpoint /path/to/action_value_ckpt \
  --output_dir /path/to/merged_ckpt \
  --copy_prefix action_ --copy_prefix value_
```

Then:

```bash
bash scripts/train_stage3.sh configs/paper/stage3_action_value_paper.yaml
```

Checkpoints are written every `steps_to_save` steps under `output_dir`; the
stage-3 checkpoint is used for serving.

---

## 8. Inference / serving

Start the policy server:

```bash
python scripts/serve_policy.py \
  --mode value_rerank \
  --selection_mode argmax \
  --num_candidates 3 \
  --config_file configs/paper/stage3_action_value_paper.yaml \
  --checkpoint_path /path/to/stage3/step_XXXXX/diffusion_pytorch_model.safetensors \
  --device cuda:0 \
  --port 18099
```

`--selection_mode argmax` is the paper mode: three candidate latent futures are
generated in one batch, each is scored once, and the arg-max candidate's latent
and value states are used by the Action Expert. The legacy
`--selection_mode elite` path performs one round of elite reweighting and is
kept for backward compatibility. Add `--return_video` to include decoded
predicted future frames in the `/act` response; it is off by default and is
used only for qualitative videos. The same response also exposes per-view
`goal_heatmaps` when available.

A synthetic request test (after the server prints `ready`):

```bash
python scripts/smoke_test_argmax.py
```

The `/act` payload format is documented in the script: `frames` is a list of
history timesteps, each with four base64-encoded camera images; `prompt` is the
language goal; `poses` and `pose_steps` are optional pose history.

---

## 9. Closed-loop simulator evaluation

`scripts/eval_sim.py` evaluates the served policy in the AerialDojo /
ProjectAirSim simulator. Point `--sim-repo` and `--task-file` at the public
AerialDojo release (Section 10).

```bash
python scripts/eval_sim.py \
  --sim-repo /path/to/AerialDojo \
  --task-file /path/to/tasks.json \
  --policy-url http://127.0.0.1:18099 \
  --seeds 42 43 44 \
  --success-distance 3.0 \
  --max-actions 150 \
  --output-jsonl eval_aerialdojo.jsonl
```

- The episode count is determined by `--task-file`; `--episodes 0` evaluates
  every task. The script does not hardcode 300 episodes.
- Set `--success-distance` to the benchmark threshold; UAV-ON and AerialDojo
  use different radii.
- SR, OSR, and DTS are computed from the simulator info. Per-step frames can be
  saved with `--save-frames-dir`.
- The simulator is an external dependency and is not part of this release.

---

## 10. Datasets and base models

The pipeline runs on publicly released resources. Download them and place them
as described in Sections 3 and 4.

| Resource | Where to obtain it |
|---|---|
| AerialDojo scenes and simulator | the public AerialDojo release, which bundles the ProjectAirSim plugin and the task files |
| UAV-ON | the public UAV-ON benchmark release; Section 3 gives the LeRobot-style layout the loader expects |
| LTX-Video, T5, VAE weights | the official LTX-Video release; set `diffusion_model.model_path` and the text-encoder / VAE paths in the configs |
| Grounding DINO teacher | the public Grounding DINO checkpoint; Section 4 lists the files the label script expects |

The same pipeline is used for both benchmarks: the scripts below run the
AerialDojo benchmark, and UAV-ON follows the same steps with its own task file
(Section 9). The configs follow the paper's reported settings.

---

## 11. Verification

The following checks passed on a 4×A100 node:

- all Python files compile;
- a 2-step stage-1 run trains, logs video/goal-heatmap losses, and saves a
  checkpoint;
- a 2-step stage-2b run trains the value expert on the success/failure mix and saves a
  checkpoint;
- a 2-step stage-3 run trains the action/value heads and saves a checkpoint;
- the serving path with `--selection_mode argmax --num_candidates 3` returns
  three candidate scores and an action over HTTP.

These are plumbing checks, not reproductions of the paper tables.

---

## Acknowledgements

This code builds on Genie-Envisioner and LTX-Video, and uses Grounding DINO. Please follow the licenses of the upstream projects.