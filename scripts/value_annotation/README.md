# Stage-aware value annotation

`annotate_value_v2.py` is the generator that produced the `state_value` and
`value_phase` columns used by the paper's AerialDojo training data
(`drone_lerobot_dataset_fantasy_v1_value_v2_full`). `batch_v2_value.py` applies
it to a LeRobot-style dataset (grounding `visible`/`conf` labels are read from
`labels_object_name_soft_top3_full`).

The scripts in `scripts/` named `annotate_drone_grounding_value.py` and
`generate_value.py` are **not** the stage-aware generator: the former adds
grounding columns, and the latter writes the older single-stage progress value.

## Stage boundary

Let $b_t = \max_v b_t^{(v)}$ be the multi-view target visibility at frame $t$.
The approach stage starts at the first frame of two consecutive visible frames:

$$
t_g=\min\{t: b_t=1,\ b_{t+1}=1\}.
$$

If a successful trajectory never has two consecutive visible frames,
`success_driven=True` sets the whole trajectory to the approach curve.

## Exploration score

For $t<t_g$:

$$
e_t=\operatorname{clip}\left(\frac{m_t+\rho_t+i_t-s_t}{3},\ 0,\ 1\right),
\qquad
y_t^* = \beta\, e_t,\quad \beta = 0.4 .
$$

with

- $m_t=\|\Delta p_t\|_2 / \max(\max_j\|\Delta p_j\|_2, 0.1)$: normalized 3D motion;
- $\rho_t=\min(|\Delta\psi_t|/(\pi/12), 1)$: rotation relative to a 15° step,
  where $\psi_t$ is the yaw of the pose quaternion $(w,x,y,z)$;
- $i_t=[c_t-c_{t-1}]_+ / \max(\max_j[c_j-c_{j-1}]_+, 10^{-6})$: positive
  grounding-confidence gain;
- $s_t=\operatorname{clip}(\text{consecutive idle frames}/3, 0, 1)$, where a
  frame is idle when both motion and rotation are close to zero.

Each signal is normalized to $[0,1]$ before the combination, and the sum is
divided by the total positive weight (3), so $e_t$ is a normalized exploration
score in $[0,1]$ and $y_t^*$ lies in $[0,\beta]=[0,0.4]$. Exploration therefore
stays at or below the approach floor and never enters the approach range
$(0.4,1]$.

## Approach score

For $t\ge t_g$:

$$
y_t^* = 0.4 + 0.6\operatorname{clip}\left(1-\frac{d_t}{\max(d_{t_g},10^{-3})},0,1\right),
$$

which lies in $[0.4, 1]$.

## Usage

```bash
python scripts/value_annotation/batch_v2_value.py \
  --root /path/to/drone_lerobot_dataset_fantasy_v1 \
  --out /path/to/drone_lerobot_dataset_fantasy_v1_value_v2_full \
  --labels /path/to/labels \
  --annotation-json /path/to/Object_annotation.json \
  --default-success
```

`--annotation-json` is only needed when the parquet has no `goal_anchor`
column. Drop `--default-success` for failure rollouts that carry
`is_success = 0`.

Grounding labels are matched by episode index, so generate them for the dataset
you are annotating. If a rollout pool has no labels of its own, pass
`--labels ""`: those trajectories then receive exploration labels throughout,
and a label file whose length does not match the episode is ignored.

For unsuccessful rollouts the exploration labels are multiplied by
`failure_scale`, while the approach labels keep the 0.4 floor and only their
distance-progress range is scaled, so they top out at
`0.4 + 0.6 * failure_scale` (0.7 with the default 0.5).

Dependencies: `numpy`, `pandas`, `pyarrow`. The grounding labels are read from
`--labels` (default `/path/to/data/labels_object_name_soft_top3_full`).