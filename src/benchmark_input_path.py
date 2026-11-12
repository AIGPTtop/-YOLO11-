"""Input-path comparison for TensorRT inference: file path vs numpy vs CUDA tensor.

Motivation: the E2E decomposition showed preprocessing dominates. This script
removes that bottleneck step by step and measures what is left.

Usage:
    python src/benchmark_input_path.py --images 200 --rounds 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def stats(model, srcs, imgsz, max_det):
    pre, inf, post, e2e = [], [], [], []
    for s in srcs:
        t0 = time.perf_counter()
        r = model.predict(s, imgsz=imgsz, conf=0.25, max_det=max_det, device="0", verbose=False)[0]
        e2e.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
    return {"pre": statistics.mean(pre), "inf": statistics.mean(inf),
            "post": statistics.mean(post), "e2e": statistics.mean(e2e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "benchmark_input_path.json")
    args = ap.parse_args()

    files = [str(p) for p in sorted((PROJECT_ROOT / "datasets/VisDrone/val/images").glob("*.jpg"))[: args.images]]

    def letterbox(im):
        h, w = im.shape[:2]
        r = min(args.imgsz / h, args.imgsz / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        out = np.full((args.imgsz, args.imgsz, 3), 114, dtype=np.uint8)
        out[:nh, :nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return out

    arrs = [letterbox(cv2.imread(f)) for f in files]
    tensors = [torch.from_numpy(a[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda() for a in arrs]

    engines = {"fp16_stock": PROJECT_ROOT / "weights/best.engine",
               "fp16_nms": PROJECT_ROOT / "weights/best_nms.engine"}
    models = {k: YOLO(str(v)) for k, v in engines.items() if v.exists()}

    jobs = []
    for name, m in models.items():
        jobs.append((f"{name}|path", m, files, 1000))
        jobs.append((f"{name}|ndarray", m, arrs, 1000))
        jobs.append((f"{name}|cuda_tensor", m, tensors, 1000))

    for _, m, srcs, md in jobs:
        stats(m, srcs[:5], args.imgsz, md)  # warmup

    acc = {name: [] for name, *_ in jobs}
    for _ in range(args.rounds):
        for name, m, srcs, md in jobs:
            acc[name].append(stats(m, srcs, args.imgsz, md))

    summary = {n: {k: statistics.median(r[k] for r in rs) for k in rs[0]} for n, rs in acc.items()}
    print(f"{'variant':28s} {'pre':>7s} {'inf':>7s} {'post':>7s} {'e2e':>8s} {'FPS':>7s}")
    for n, s in summary.items():
        print(f"{n:28s} {s['pre']:7.2f} {s['inf']:7.2f} {s['post']:7.2f} {s['e2e']:8.2f} {1000/s['e2e']:7.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
