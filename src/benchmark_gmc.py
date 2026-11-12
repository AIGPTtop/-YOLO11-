"""Micro-benchmark of Ultralytics GMC (camera motion compensation) variants.

Runs on CPU only (cv2), independent of GPU contention, so the numbers are stable
while other jobs occupy the GPUs.

Usage:
    python src/benchmark_gmc.py --frames 150
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIGS = [
    ("sparseOptFlow", 2),
    ("sparseOptFlow", 4),
    ("sparseOptFlow", 6),
    ("orb", 2),
    ("ecc", 2),
    ("none", 2),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="uav0000305_00000_v")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "gmc_benchmark.json")
    args = ap.parse_args()

    from ultralytics.trackers.utils.gmc import GMC

    seq_dir = PROJECT_ROOT / "datasets" / "VisDrone2019-MOT-val" / "sequences" / args.seq
    paths = sorted(seq_dir.glob("*.jpg"))[: args.frames]
    frames = [cv2.imread(str(p)) for p in paths]
    frames = [f for f in frames if f is not None]
    print(f"{len(frames)} frames  {frames[0].shape[1]}x{frames[0].shape[0]}")

    rows = []
    for method, ds in CONFIGS:
        gmc = GMC(method=method, downscale=ds)
        gmc.apply(frames[0])  # warmup / init
        t0 = time.perf_counter()
        for f in frames[1:]:
            gmc.apply(f)
        dt = (time.perf_counter() - t0) / max(1, len(frames) - 1) * 1000
        rows.append({"method": method, "downscale": ds, "ms_per_frame": round(dt, 3)})
        print(f"{method:14s} ds={ds}  {dt:7.2f} ms/frame  ({1000/dt:6.1f} FPS ceiling)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
