"""Measure the detection+tracking pipeline on different backends.

This is the link the project was missing: TensorRT was only ever benchmarked on
single images, never inside the tracking loop.

Usage:
    python src/benchmark_tracking_backend.py --seq uav0000305_00000_v \
        --models pt engine --trackers bytetrack botsort_gmc --frames 150
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "pt": PROJECT_ROOT / "weights" / "best.pt",
    "engine": PROJECT_ROOT / "weights" / "best.engine",
    "engine_nms": PROJECT_ROOT / "weights" / "best_nms.engine",
}
TRACKERS = {
    "bytetrack": "bytetrack.yaml",
    "botsort": str(PROJECT_ROOT / "configs" / "botsort_visdrone_gmc.yaml"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="uav0000305_00000_v")
    ap.add_argument("--models", nargs="+", default=["pt", "engine"])
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "botsort"])
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "tracking_backend.json")
    args = ap.parse_args()

    seq = PROJECT_ROOT / "datasets" / "VisDrone2019-MOT-val" / "sequences" / args.seq
    paths = sorted(seq.glob("*.jpg"))[: args.frames + args.warmup]
    import cv2

    frames = [cv2.imread(str(p)) for p in paths]

    rows = []
    for mname in args.models:
        mpath = MODELS[mname]
        if not mpath.exists():
            print(f"skip {mname}: {mpath} missing")
            continue
        for tname in args.trackers:
            model = YOLO(str(mpath))
            lat = []
            for i, f in enumerate(frames):
                t0 = time.perf_counter()
                model.track(f, persist=True, tracker=TRACKERS[tname], imgsz=args.imgsz,
                            conf=args.conf, device="0", verbose=False)
                if i >= args.warmup:
                    lat.append((time.perf_counter() - t0) * 1000)
            mean = statistics.mean(lat)
            p95 = sorted(lat)[int(len(lat) * 0.95)]
            rows.append({"model": mname, "tracker": tname, "mean_ms": round(mean, 2),
                         "p95_ms": round(p95, 2), "fps": round(1000 / mean, 1)})
            print(f"{mname:10s} + {tname:10s}  {mean:7.2f} ms  {1000/mean:6.1f} FPS   p95={p95:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
