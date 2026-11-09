#!/usr/bin/env python3
"""Batch-annotate fantasy_v1 + collect data with v2 two-stage value.

Writes parquet with `state_value` column replaced by v2 value (0..1) and keeps
all other columns. Non-success episodes have explore phase only; fantasy_v1 has
no is_success so all treated as success (expert search trajectories).
"""
import sys, argparse, glob, os, shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "/tmp")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "av2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotate_value_v2.py")
)
av2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(av2)


def write_parquet_v2(src_path, out_path, value, phase, ep, is_success, goal):
    t = pq.read_table(src_path)
    cols = {c: t.column(c).to_pylist() for c in t.column_names}
    T = len(value)
    # state_value as list<double> rows of len 1 (v1 encoded scalar per frame)
    sv = [[float(v)] for v in value]
    cols["state_value"] = sv
    cols["value_phase"] = [int(p) for p in phase]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def arr(col, name):
        v = cols[col]
        if name in ("state_value", "value_phase", "dist_to_goal"):
            return pa.array(v)
        return pa.array(v)

    # rebuild typed columns like source to keep pyarrow happy
    fields = {}
    for c in t.column_names:
        col = t.column(c)
        fields[c] = pa.array(cols[c], type=col.type)
    # add phase
    fields = dict(fields)
    # state_value was replaced already via cols; rebuild with its type
    sv_type = t.column("state_value").type if "state_value" in t.column_names else pa.list_(pa.float64())
    fields["state_value"] = pa.array(sv, type=sv_type)
    fields["value_phase"] = pa.array([int(p) for p in phase], type=pa.int64())
    pq.write_table(pa.table(fields), str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", default="/path/to/data/labels_object_name_soft_top3_full")
    ap.add_argument("--annotation-json", default="",
                    help="Object annotation JSON with description + Object_anchor_position; "
                         "only needed when the dataset has no goal_anchor column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--default-success", action="store_true",
                    help="dataset has no is_success column; all eps are success "
                         "(e.g. expert A* search trajectories)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta_src = os.path.join(args.root, "meta")
    meta_dst = os.path.join(args.out, "meta")
    if os.path.isdir(meta_src) and not os.path.exists(meta_dst):
        shutil.copytree(meta_src, meta_dst)
    videos_src = os.path.join(args.root, "videos")
    videos_dst = os.path.join(args.out, "videos")
    if os.path.isdir(videos_src) and not os.path.exists(videos_dst):
        os.symlink(os.path.abspath(videos_src), videos_dst)

    tasks, resolve = av2._load_tasks_and_goal_lookup(args.root, args.annotation_json or None)
    files = sorted(glob.glob(os.path.join(args.root, "data/chunk-*/episode_*.parquet")))
    if args.limit:
        files = files[:args.limit]

    report = []
    for i, f in enumerate(files):
        try:
            v, ph, ep, succ, goal = av2.annotate_one_v2(
                f, args.labels, tasks, resolve, default_success=args.default_success)
            rel = os.path.relpath(f, args.root)
            out_path = os.path.join(args.out, rel)
            write_parquet_v2(f, out_path, v, ph, ep, succ, goal)
            report.append((ep, bool(succ), float(v.mean()), float(v.max())))
            if i % 200 == 0:
                print(f"[{i}/{len(files)}] ep{ep} succ={succ} vmean={v.mean():.3f} vmax={v.max():.3f}", flush=True)
        except Exception as e:
            print(f"[{i}] FAIL {f}: {e}", flush=True)
    print(f"DONE {len(report)}/{(len(files))}")
    if report:
        arr = np.array(report)
        ok = arr[arr[:, 1].astype(bool)] if len(arr) else arr
        print(f"samples={len(report)} success_frac={arr[:,1].mean():.3f} "
              f"avg_vmean={arr[:,2].mean():.3f} max_vmax={arr[:,3].max():.3f}")


if __name__ == "__main__":
    main()