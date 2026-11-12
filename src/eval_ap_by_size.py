"""COCO-style AP evaluation with size buckets and per-class breakdown.

Used to answer "does the P2 branch actually help *small* objects?", which the
aggregate mAP cannot show on VisDrone.

Usage:
    python src/eval_ap_by_size.py --model weights/best.pt --imgsz 960 --tag E3
    python src/eval_ap_by_size.py --model outputs/train/yolo11n_visdrone_baseline/weights/best.pt \
        --imgsz 640 --tag E0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# COCO area ranges in pixels^2
AREA_RANGES = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, 1e10),
}

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def load_gt(labels_dir: Path, images_dir: Path):
    """Return {stem: (boxes_xyxy, classes, areas)} in original image pixels."""
    gts = {}
    for lbl in sorted(labels_dir.glob("*.txt")):
        img_path = images_dir / f"{lbl.stem}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        rows = np.loadtxt(lbl, ndmin=2)
        if rows.size == 0:
            gts[lbl.stem] = (np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0))
            continue
        cls = rows[:, 0].astype(int)
        cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        gts[lbl.stem] = (boxes, cls, bw * bh)
    return gts


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def average_precision(recall, precision):
    """COCO 101-point interpolation."""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(mrec, grid, side="left")
    idx = np.clip(idx, 0, len(mpre) - 1)
    return float(np.mean(mpre[idx]))


def evaluate(gts, preds, num_classes=10, area=None):
    """Return (AP, AP50, AP75, per_class_ap) for one area range."""
    lo, hi = area if area else (0.0, 1e10)
    aps = np.full((num_classes, len(IOU_THRESHOLDS)), np.nan)
    empty = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros(0))

    for c in range(num_classes):
        # per-threshold running lists of (score, is_tp)
        tp = [[] for _ in IOU_THRESHOLDS]
        sc = [[] for _ in IOU_THRESHOLDS]
        n_gt = 0

        for stem, (gboxes, gcls, garea) in gts.items():
            m = (gcls == c) & (garea >= lo) & (garea < hi)
            g = gboxes[m]
            n_gt += len(g)

            p_boxes, p_scores, p_cls, p_area = preds.get(stem, empty)
            pm = (p_cls == c) & (p_area >= lo) & (p_area < hi)
            p = p_boxes[pm]
            s = p_scores[pm]
            if len(s) == 0:
                continue
            order = np.argsort(-s)
            p, s = p[order], s[order]
            ious = iou_matrix(p, g) if len(g) else np.zeros((len(p), 0))

            for ti, thr in enumerate(IOU_THRESHOLDS):
                if len(g) == 0:
                    tp[ti].extend([0] * len(p))
                    sc[ti].extend(s.tolist())
                    continue
                matched = np.zeros(len(g), dtype=bool)
                for di in range(len(p)):
                    row = ious[di].copy()
                    if matched.any():
                        row[matched] = -1.0
                    j = int(np.argmax(row))
                    if row[j] >= thr:
                        matched[j] = True
                        tp[ti].append(1)
                    else:
                        tp[ti].append(0)
                    sc[ti].append(float(s[di]))

        for ti in range(len(IOU_THRESHOLDS)):
            if n_gt == 0:
                continue
            scores = np.array(sc[ti])
            tpv = np.array(tp[ti])
            if len(scores) == 0:
                aps[c, ti] = 0.0
                continue
            order = np.argsort(-scores)
            tpv = tpv[order]
            ctp = np.cumsum(tpv)
            cfp = np.cumsum(1 - tpv)
            recall = ctp / (n_gt + 1e-9)
            precision = ctp / (ctp + cfp + 1e-9)
            aps[c, ti] = average_precision(recall, precision)

    per_class = np.nanmean(aps, axis=1)
    return {
        "AP50-95": float(np.nanmean(aps)),
        "AP50": float(np.nanmean(aps[:, 0])),
        "AP75": float(np.nanmean(aps[:, 5])),
        "per_class": {CLASS_NAMES[i]: float(per_class[i]) for i in range(num_classes)},
    }


def predict_all(model, images_dir: Path, imgsz: int, device="0"):
    preds = {}
    for img_path in sorted(images_dir.glob("*.jpg")):
        r = model.predict(str(img_path), imgsz=imgsz, conf=0.001, iou=0.7,
                          max_det=1000, device=device, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            preds[img_path.stem] = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros(0))
            continue
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        preds[img_path.stem] = (xyxy, conf, cls, area)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results")
    args = ap.parse_args()

    images = PROJECT_ROOT / "datasets" / "VisDrone" / "val" / "images"
    labels = PROJECT_ROOT / "datasets" / "VisDrone" / "val" / "labels"

    print(f"[{args.tag}] loading GT ...")
    gts = load_gt(labels, images)
    print(f"[{args.tag}] {len(gts)} images, running inference ...")
    preds = predict_all(YOLO(str(args.model)), images, args.imgsz, args.device)

    out = {"tag": args.tag, "model": str(args.model), "imgsz": args.imgsz, "buckets": {}}
    for name, rng in AREA_RANGES.items():
        print(f"[{args.tag}] AP @ {name} ...")
        out["buckets"][name] = evaluate(gts, preds, area=rng)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"ap_by_size_{args.tag}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: {m: round(v[m], 4) for m in ("AP50-95", "AP50", "AP75")} for k, v in out["buckets"].items()}, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
