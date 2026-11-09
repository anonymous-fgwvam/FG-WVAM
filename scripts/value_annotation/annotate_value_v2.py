#!/usr/bin/env python3
"""v2 two-stage value annotation.

phase  = explore / approach, switched by grounding `visible` (any camera),
         entering approach at first 2-consecutive-visible frame (no fallback).

explore_t   = beta * clip((m_t + rho_t + i_t - s_t) / 3, 0, 1),  beta = 0.4
approach_t  = 0.4 + 0.6 * clip(1 - d_t/d_{t*}, 0, 1)

Each signal is normalized to [0, 1] first, combined with equal weights, and
divided by the sum of the absolute weights. The result is a normalized
exploration score e_t in [0, 1] scaled by the stage boundary beta = 0.4.
Exploration scores therefore lie in [0, 0.4], so the exploration ceiling is
the stage boundary beta and never exceeds the approach range [0.4, 1].

failed episodes: exploration frames scaled by failure_scale (default 0.5);
                 approach frames scaled by (0.4 + 0.6*failure_scale).

Signals (per frame t, from poses [T,7] and grounding heatmaps):
  m_t = |pos_t - pos_{t-1}| normalized by the largest step in the episode
  rho_t = |yaw_t - yaw_{t-1}| / 15deg, with yaw read from the pose quaternion
  i_t = [max(conf_t) - max(conf_{t-1})]_+ normalized by its own maximum
  s_t = clip(consecutive near-static frames / 3, 0, 1), where near-static
        means both motion and rotation are close to zero
"""
import argparse, json, glob, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

# Equal weights on the normalized signals. The combination is normalized by the
# sum of the three positive weights (motion, rotation, confidence gain) and then
# scaled by the stage boundary beta, so the exploration score lies in
# [0, beta] and never exceeds the approach floor.
W_MOTION, W_ROTATION, W_IGAIN, W_STATIC = 1.0, 1.0, 1.0, 1.0
EXPLORE_NORM = W_MOTION + W_ROTATION + W_IGAIN
EXPLORE_BETA = 0.4          # stage boundary; exploration scores lie in [0, beta]
EXPLORE_CAP = EXPLORE_BETA  # kept as the upper bound of the exploration range
ROTATION_REF = np.pi / 12   # 15 degrees per decision step
APPROACH_BASE = 0.4
APPROACH_RANGE = 0.6
FAIL_SCALE = 0.5
CONSECUTIVE_VISIBLE = 2


def dist_to_goal(poses):  # [T,3] -> [T]
    return np.linalg.norm(poses[:, :3], axis=-1)


def _yaw_from_pose(poses):
    """Heading angle per frame, in radians.

    Uses the pose quaternion (w, x, y, z) when the pose carries an
    orientation; falls back to the direction of the XY displacement
    otherwise.
    """
    T = len(poses)
    if poses.shape[1] >= 7:
        q = poses[:, 3:7].astype(np.float64)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return np.unwrap(np.arctan2(2.0 * (w * z + x * y),
                                    1.0 - 2.0 * (y * y + z * z)))
    if T < 2:
        return np.zeros(T)
    d = np.diff(poses[:, :3].astype(np.float64), axis=0)
    return np.concatenate([[0.0], np.unwrap(np.arctan2(d[:, 1], d[:, 0]))])


def compute_phase_and_value(poses, goal, visible, conf, success_driven=False):
    """visible: [n_cam, T] or None. conf: [n_cam,T] or None.
    success_driven=True: when no grounding signal present, encode "this
    episode reaches the goal" via d->1 curve (used for collected success
    episodes that lack grounding). Returns value[T], phase[T]."""
    T = len(poses)
    value = np.zeros(T, dtype=np.float64)
    phase = np.zeros(T, dtype=np.int64)

    # ---- signals ----
    pos = poses[:, :3].astype(np.float64)
    yaw = _yaw_from_pose(poses)
    d = np.linalg.norm(pos - goal.reshape(3), axis=-1)          # [T]

    add = np.zeros(T)
    rot = np.zeros(T)
    ig  = np.zeros(T)
    if T > 1:
        add[1:] = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)   # meters moved
        rot[1:] = np.minimum(np.abs(np.diff(yaw)) / ROTATION_REF, 1.0)
    if conf is not None:
        c = np.asarray(conf, dtype=np.float64)                   # [n_cam,T]
        mx = c.max(axis=0)
        ig[1:] = np.clip(mx[1:] - mx[:-1], 0, None)              # conf gain
        ig = ig / max(ig.max(), 1e-6)
    stop = ((add < 0.05) & (rot < 0.05))                          # neither moves nor turns
    streak = np.zeros(T, dtype=np.int64)
    for t in range(T):
        if stop[t]:
            streak[t] = streak[t-1] + 1 if t > 0 else 1
    repeat = np.clip(streak / 3.0, 0, 1.0)                        # 3+ idle frames -> 1

    motion = add / max(add.max(), 0.1)                            # [0, 1]
    score = (W_MOTION * motion + W_ROTATION * rot + W_IGAIN * ig
             - W_STATIC * repeat) / EXPLORE_NORM                 # normalized score
    explore = EXPLORE_BETA * np.clip(score, 0.0, 1.0)            # [0, beta]

    # ---- phase switch ----
    anyvis = np.zeros(T, dtype=bool)
    if visible is not None:
        anyvis = (np.asarray(visible).sum(axis=0) > 0)
    t_star = None
    if anyvis.any():
        cnt = 0
        for t in range(T):
            if anyvis[t]:
                cnt += 1
                if cnt >= CONSECUTIVE_VISIBLE:
                    t_star = t - (CONSECUTIVE_VISIBLE - 1)
                    break
            else:
                cnt = 0
    if t_star is None:
        if success_driven and goal is not None:
            # no grounding, but episode is known-success: use whole-episode
            # distance progress as the approach curve (0.4 -> 1 if reaches).
            d0 = max(float(d[0]), 1e-3)
            prog = np.clip(1.0 - d / d0, 0, 1)
            return APPROACH_BASE + APPROACH_RANGE * prog, np.ones(T, dtype=np.int64)
        value = explore                      # all explore
        return value, np.zeros(T, dtype=np.int64)
    # approach from t_star
    value[:t_star] = explore[:t_star]
    phase[:t_star] = 0
    d_star = max(float(d[t_star]), 1e-3)
    for t in range(t_star, T):
        prog = np.clip(1.0 - d[t]/d_star, 0, 1)
        value[t] = APPROACH_BASE + APPROACH_RANGE*prog
        phase[t] = 1
    return value, phase


def load_grounding(ep_idx, gl_dir):
    p = os.path.join(gl_dir, f"labels_ep{ep_idx:06d}.npz")
    if not os.path.exists(p):
        return None, None
    d = np.load(p)
    vis = d["visible"]
    conf = d["conf"]
    return vis, conf


def annotate_one(parquet_path, gl_dir, out_path, failure_scale=FAIL_SCALE):
    t = pq.read_table(parquet_path)
    poses = np.stack(t.column("observation.pose").to_pylist()).astype(np.float64)  # [T,7]
    ep = int(t.column("episode_index")[0].as_py())
    df_cols = {c: t.column(c).to_pylist() for c in t.column_names}

    goal = None
    if "goal_anchor" in df_cols and df_cols["goal_anchor"]:
        goal = np.asarray(df_cols["goal_anchor"][0], dtype=np.float64)
    if goal is None and "dist_to_goal" in df_cols:
        goal = poses[-1, :3]  # fallback: last pose
    is_success = int(df_cols.get("is_success", [0])[0])
    vis, conf = load_grounding(ep, gl_dir)
    if vis is None:
        vis, conf = None, None
    value, phase = compute_phase_and_value(poses, goal, vis, conf)
    if not is_success:
        value = value * failure_scale
    return value, phase, ep, is_success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", default="/path/to/data/labels_object_name_soft_top3_full")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="annotation_v2.stage1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.root, "data/chunk-*/episode_*.parquet")))
    if args.limit: files = files[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    for i, f in enumerate(files):
        try:
            value, phase, ep, succ = annotate_one(f, args.labels, None)
            # write a small report
            with open(os.path.join(args.out, "value_report.log"), "a") as h:
                if i % 50 == 0:
                    h.write(f"[{i}/{len(files)}] ep={ep} succ={succ} v0={value[0]:.2f} vlast={value[-1]:.2f} phase_approach={phase.mean():.2f}\n")
            print(f"[{i}/{len(files)}] ep={ep} succ={succ} vmean={value.mean():.3f} vmax={value.max():.3f}", flush=True)
        except Exception as e:
            print(f"[{i}] ep err {f}: {e}", flush=True)
    print("DONE")

if __name__ == "__main__":
    main()


def _load_tasks_and_goal_lookup(root, annotation_json=None):
    import re as _re
    tasks = {}
    tp = Path(root) / "meta" / "tasks.jsonl"
    if tp.exists():
        for ln in open(tp):
            if ln.strip():
                o = json.loads(ln)
                tasks[int(o["task_index"])] = o["task"]
    lookup = {}
    if annotation_json:
        anno = json.load(open(annotation_json))
        for a in anno:
            desc = a.get("description") or ""
            stripped = _re.sub(r"^(A|An)\s+", "", desc, flags=_re.IGNORECASE)
            if stripped:
                lookup[stripped] = np.asarray(a.get("Object_anchor_position", [0,0,0]), dtype=np.float64)
    def resolve(task_text):
        t = _re.sub(r"^navigate to the\s+", "", task_text or "", flags=_re.IGNORECASE).strip()
        return lookup.get(t)
    return tasks, resolve


def annotate_one_v2(parquet_path, gl_dir, tasks, resolve_goal, failure_scale=FAIL_SCALE, default_success=False):
    t = pq.read_table(parquet_path)
    poses = np.stack([np.asarray(x, dtype=np.float64) for x in t.column("observation.pose").to_pylist()])
    ep = int(t.column("episode_index")[0].as_py())
    goal = None
    if "goal_anchor" in t.column_names:
        ga = t.column("goal_anchor")[0].as_py()
        if ga is not None:
            goal = np.asarray(ga, dtype=np.float64)
    if goal is None:
        ti = int(t.column("task_index")[0].as_py()) if "task_index" in t.column_names else 0
        goal = resolve_goal(tasks.get(ti, ""))
    if goal is None:
        goal = poses[-1, :3]  # last-pose fallback
    if "is_success" in t.column_names:
        is_success = int(t.column("is_success")[0].as_py())
    else:
        is_success = int(default_success)   # expert data: no column -> all success
    vis, conf = load_grounding(ep, gl_dir)
    if vis is not None and np.shape(vis)[1] != len(poses):
        # The label file belongs to a different dataset with the same episode
        # index; fall back to exploration labels instead of mixing episodes.
        vis, conf = None, None
    value, phase = compute_phase_and_value(poses, goal, vis, conf, success_driven=bool(is_success))
    # Failure handling: lower the exploration phase and shrink the progress
    # range of the approach phase.
    if not is_success:
        # Exploration: halve the search credit, so failed searching counts less.
        mask_explore = (phase == 0)
        value[mask_explore] = np.clip(value[mask_explore] * failure_scale, 0, EXPLORE_CAP)
        # Approach: the target was still found, so keep the 0.4 floor and shrink
        # only the distance-progress range. A failed trajectory therefore tops
        # out at APPROACH_BASE + APPROACH_RANGE * failure_scale (= 0.7) instead
        # of falling below the exploration ceiling.
        mask_approach = ~mask_explore
        value[mask_approach] = APPROACH_BASE + failure_scale * (value[mask_approach] - APPROACH_BASE)
    return value, phase, ep, is_success, goal
