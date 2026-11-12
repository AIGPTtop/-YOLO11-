"""Build a failure-case figure: GT (green) vs prediction (red) on hard frames.

The densest VisDrone MOT-val sequence (uav0000268_05773_v, 978 frames) is where
the detector drops to ~21% recall, so it makes the most honest error panel.

Usage:
    python src/make_failure_cases.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van",
               "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def load_gt(ann: Path):
    rows = np.loadtxt(ann, delimiter=",", ndmin=2)
    gt = {}
    for r in rows:
        cls = int(r[7])
        if cls < 1 or cls > 10:
            continue
        f = int(r[0])
        gt.setdefault(f, []).append((r[2:6].astype(float), cls))
    return gt


def iou(a, b):
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[0] * wh[1]
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def score_frame(gt_boxes, pred_boxes):
    if len(gt_boxes) == 0:
        return 0
    matched = 0
    for g in gt_boxes:
        if len(pred_boxes) and max(iou(g, p) for p in pred_boxes) >= 0.5:
            matched += 1
    return len(gt_boxes) - matched  # missed targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="uav0000268_05773_v")
    ap.add_argument("--model", type=Path, default=PROJECT_ROOT / "weights" / "best.pt")
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "assets" / "failure_cases.jpg")
    args = ap.parse_args()

    seq_dir = PROJECT_ROOT / "datasets" / "VisDrone2019-MOT-val" / "sequences" / args.seq
    ann = PROJECT_ROOT / "datasets" / "VisDrone2019-MOT-val" / "annotations" / f"{args.seq}.txt"
    gt = load_gt(ann)
    model = YOLO(str(args.model))

    frames = sorted(seq_dir.glob("*.jpg"))
    picks = []
    for i in range(0, len(frames), args.stride):
        f = i + 1
        img = cv2.imread(str(frames[i]))
        r = model.predict(img, imgsz=960, conf=0.25, max_det=1000, device="0", verbose=False)[0]
        preds = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
        gt_boxes = [b for b, _ in gt.get(f, [])]
        miss = score_frame(gt_boxes, preds)
        picks.append((miss, f, img, preds, gt_boxes, len(gt_boxes)))
        print(f"frame {f:4d}  gt={len(gt_boxes):3d}  pred={len(preds):3d}  missed={miss:3d}")

    picks.sort(key=lambda x: -x[0])
    selected = picks[: args.top]
    panels = []
    for miss, f, img, preds, gt_boxes, n_gt in selected:
        vis = img.copy()
        for b in gt_boxes:
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[0] + b[2]), int(b[1] + b[3])), (0, 255, 0), 2)
        for p in preds:
            cv2.rectangle(vis, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 255), 2)
        h, w = vis.shape[:2]
        scale = 900 / w
        vis = cv2.resize(vis, (900, int(h * scale)))
        cv2.rectangle(vis, (0, 0), (900, 34), (0, 0, 0), -1)
        cv2.putText(vis, f"{args.seq}  frame {f}  GT {n_gt}  pred {len(preds)}  missed {miss}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(vis)

    out = np.vstack(panels)
    cv2.putText(out, "green = GT   red = prediction", (10, out.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), out, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
