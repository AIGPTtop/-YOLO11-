"""Throughput scaling with batch size (PyTorch backend).

The TensorRT engines in this repo are exported with batch=1, so a fair batch>1
throughput measurement has to use the PyTorch model. Reported in images/s.

Usage:
    python src/benchmark_batch_throughput.py --batches 1 2 4 8 16 --images 64 --rounds 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "batch_throughput.json")
    args = ap.parse_args()

    img_dir = PROJECT_ROOT / "datasets" / "VisDrone" / "val" / "images"
    files = [str(p) for p in sorted(img_dir.glob("*.jpg"))[: args.images]]
    model = YOLO(str(args.model))

    # warmup
    model.predict(files[:8], imgsz=args.imgsz, batch=8, device="0", verbose=False)

    rows = []
    for bs in args.batches:
        per_round = []
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            model.predict(files, imgsz=args.imgsz, batch=bs, device="0", verbose=False)
            dt = time.perf_counter() - t0
            per_round.append(len(files) / dt)
        ips = statistics.median(per_round)
        rows.append({"batch": bs, "images_per_s": round(ips, 1), "ms_per_image": round(1000 / ips, 2)})
        print(f"batch={bs:3d}  {ips:7.1f} img/s   {1000/ips:6.2f} ms/img")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": "best.pt", "imgsz": args.imgsz, "rows": rows}, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
