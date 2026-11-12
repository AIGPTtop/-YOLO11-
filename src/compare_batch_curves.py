"""Compare training curves of two runs epoch-by-epoch.

Used for the batch-size sensitivity check: E0 (batch=16 @640) vs batch=32 @640.
Comparing at matched epochs is a cleaner control than comparing final values,
because early stopping can land on different epochs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        r = {k.strip(): v for k, v in r.items()}
        out[int(float(r["epoch"]))] = {
            "map50": float(r.get("metrics/mAP50(B)", 0)),
            "map": float(r.get("metrics/mAP50-95(B)", 0)),
            "p": float(r.get("metrics/precision(B)", 0)),
            "r": float(r.get("metrics/recall(B)", 0)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, default=PROJECT_ROOT / "outputs/train/yolo11n_visdrone_baseline/results.csv")
    ap.add_argument("--b", type=Path, default=PROJECT_ROOT / "outputs/train/yolo11n_visdrone_baseline_b32/results.csv")
    ap.add_argument("--label-a", default="batch16")
    ap.add_argument("--label-b", default="batch32")
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    common = sorted(set(a) & set(b))
    if not common:
        print("no common epochs")
        return
    print(f"{'epoch':>5} | {args.label_a:>18} | {args.label_b:>18} | delta")
    print("-" * 62)
    for e in common:
        if e % 5 and e != common[-1]:
            continue
        da = b[e]["map"] - a[e]["map"]
        print(f"{e:5d} | mAP50-95={a[e]['map']:.4f} mAP50={a[e]['map50']:.4f} | "
              f"mAP50-95={b[e]['map']:.4f} mAP50={b[e]['map50']:.4f} | {da:+.4f}")

    best_a = max(a.values(), key=lambda x: x["map"])
    best_b = max(b.values(), key=lambda x: x["map"])
    print()
    print(f"best {args.label_a}: mAP50-95={best_a['map']:.4f} mAP50={best_a['map50']:.4f}")
    print(f"best {args.label_b}: mAP50-95={best_b['map']:.4f} mAP50={best_b['map50']:.4f}")
    deltas = [b[e]["map"] - a[e]["map"] for e in common]
    print(f"mean |delta| over {len(common)} matched epochs = {sum(abs(d) for d in deltas)/len(common):.4f}")


if __name__ == "__main__":
    main()
