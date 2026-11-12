"""End-to-end latency comparison of deployment variants (interleaved A/B).

Variants
  stock      : weights/best.engine            (NMS in Python/torch)
  nms_engine : weights/best_nms.engine        (NMS fused into TensorRT)
  + max_det  : 1000 vs 300

Runs are interleaved (A,B,C,D,A,B,C,D) so that a noisy shared GPU affects all
variants equally; report deltas rather than absolute numbers if the box is busy.

Usage:
    python src/benchmark_e2e.py --runs 2 --images 200
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def measure(model, images, imgsz, max_det, conf=0.25):
    pre, inf, post, e2e = [], [], [], []
    for p in images:
        t0 = time.perf_counter()
        src = p if isinstance(p, (str, Path)) else p
        r = model.predict(src, imgsz=imgsz, conf=conf, max_det=max_det, device="0", verbose=False)[0]
        e2e.append((time.perf_counter() - t0) * 1000)
        s = r.speed
        pre.append(s["preprocess"])
        inf.append(s["inference"])
        post.append(s["postprocess"])
    return {
        "preprocess_ms": statistics.mean(pre),
        "inference_ms": statistics.mean(inf),
        "postprocess_ms": statistics.mean(post),
        "e2e_ms": statistics.mean(e2e),
        "e2e_fps": 1000 / statistics.mean(e2e),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=200)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--preresize", action="store_true", help="also time a variant fed with pre-letterboxed frames")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "benchmark_e2e.json")
    args = ap.parse_args()

    img_dir = PROJECT_ROOT / "datasets" / "VisDrone" / "val" / "images"
    images = sorted(img_dir.glob("*.jpg"))[: args.images]

    engines = {
        "stock_trt": (PROJECT_ROOT / "weights" / "best.engine", 1000),
        "nms_trt": (PROJECT_ROOT / "weights" / "best_nms.engine", 1000),
        "nms_trt_md300": (PROJECT_ROOT / "weights" / "best_nms.engine", 300),
        "stock_trt_md300": (PROJECT_ROOT / "weights" / "best.engine", 300),
        "int8_trt": (PROJECT_ROOT / "weights" / "best_int8.engine", 1000),
    }
    models = {}
    for name, (path, _) in engines.items():
        if path.exists():
            models[name] = YOLO(str(path))
    print("loaded:", list(models))

    # Optional: feed already letterboxed 960x960 frames (decode/resize done by the
    # video pipeline or an ISP). Isolates how much of E2E is Python preprocessing.
    cached = None
    if args.preresize:
        import cv2
        import numpy as np

        def letterbox(im, size):
            h, w = im.shape[:2]
            r = min(size / h, size / w)
            nw, nh = int(round(w * r)), int(round(h * r))
            resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((size, size, 3), 114, dtype=np.uint8)
            dw, dh = (size - nw) // 2, (size - nh) // 2
            canvas[dh:dh + nh, dw:dw + nw] = resized
            return canvas

        cached = [letterbox(cv2.imread(str(p)), args.imgsz) for p in images]
        print(f"pre-letterboxed {len(cached)} frames")

    jobs = [(name, models[name], engines[name][1], images) for name in models]
    if cached is not None:
        for nm in ("stock_trt", "nms_trt"):
            if nm in models:
                jobs.append((nm + "_preresized", models[nm], 1000, cached))

    acc = {name: [] for name, *_ in jobs}
    # warmup: first calls build CUDA contexts / TRT kernels and would otherwise
    # inflate the first measured variant
    for name, m, md, src in jobs:
        measure(m, src[:10], args.imgsz, md)
    print("warmup done")

    for run in range(args.runs):
        for name, m, md, src in jobs:
            r = measure(m, src, args.imgsz, md)
            acc[name].append(r)
            print(f"run{run+1} {name:20s} pre={r['preprocess_ms']:.3f} inf={r['inference_ms']:.3f} "
                  f"post={r['postprocess_ms']:.3f} e2e={r['e2e_ms']:.3f} fps={r['e2e_fps']:.1f}")

    summary = {}
    for name, runs in acc.items():
        summary[name] = {
            k: statistics.mean(r[k] for r in runs) for k in runs[0]
        }
    base = summary.get("stock_trt", {}).get("e2e_ms")
    print("\n=== summary (mean of runs) ===")
    for name, s in summary.items():
        rel = f"{(1 - s['e2e_ms'] / base) * 100:+.1f}%" if base else ""
        print(f"{name:20s} pre={s['preprocess_ms']:.3f} inf={s['inference_ms']:.3f} "
              f"post={s['postprocess_ms']:.3f} e2e={s['e2e_ms']:.3f} ms  {s['e2e_fps']:.1f} FPS  {rel}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": vars(args) | {"out": str(args.out)}, "summary": summary}, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
