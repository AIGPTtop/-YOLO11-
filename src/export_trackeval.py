"""Convert VisDrone2019-MOT-val + our tracker outputs to TrackEval (MOTChallenge) layout.

We remap all 10 VisDrone classes to MOTChallenge's single "pedestrian" class and
drop ignore-regions/others, so that both TrackEval and our own evaluator see
*identical* inputs. That isolates the metric implementation from the ignore-region
policy, which is what we want to cross-check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from evaluate_mot import evaluate  # noqa: E402


def read_txt(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "VisDrone2019-MOT-val")
    ap.add_argument("--trackers-dir", type=Path, default=PROJECT_ROOT / "outputs" / "motval_full")
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "botsort_no_gmc", "botsort_gmc"])
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "trackeval")
    ap.add_argument("--bench", default="VisDrone")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    bench_split = f"{args.bench}-{args.split}"
    gt_root = args.out / "gt" / "mot_challenge" / bench_split
    trk_root = args.out / "trackers" / "mot_challenge" / bench_split
    seqmap_dir = args.out / "gt" / "mot_challenge" / "seqmaps"
    for d in (gt_root, trk_root, seqmap_dir):
        d.mkdir(parents=True, exist_ok=True)

    seqs = sorted(p.name for p in (args.dataset / "sequences").iterdir() if p.is_dir())
    print(f"{len(seqs)} sequences")

    for seq in seqs:
        frames = sorted((args.dataset / "sequences" / seq).glob("*.jpg"))
        h, w = cv2.imread(str(frames[0])).shape[:2]
        (gt_root / seq / "gt").mkdir(parents=True, exist_ok=True)
        with open(gt_root / seq / "seqinfo.ini", "w") as f:
            f.write(
                f"[Sequence]\nseqName={seq}\nimDir=img1\nframeRate=25\n"
                f"seqLength={len(frames)}\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n"
            )
        gt = read_txt(args.dataset / "annotations" / f"{seq}.txt")
        keep = np.isin(gt[:, 7].astype(int), np.arange(1, 11))
        gt = gt[keep]
        out = np.column_stack([gt[:, 0], gt[:, 1], gt[:, 2:6], np.ones(len(gt)), np.ones(len(gt)), np.ones(len(gt))])
        np.savetxt(gt_root / seq / "gt" / "gt.txt", out, delimiter=",", fmt="%.2f")

        for trk in args.trackers:
            src = args.trackers_dir / trk / f"{seq}.txt"
            if not src.exists():
                print(f"  missing {src}")
                continue
            pred = read_txt(src)
            keep = np.isin(pred[:, 7].astype(int), np.arange(1, 11))
            pred = pred[keep]
            out = np.column_stack([pred[:, 0], pred[:, 1], pred[:, 2:6], pred[:, 6], np.ones(len(pred))])
            d = trk_root / trk / "data"
            d.mkdir(parents=True, exist_ok=True)
            np.savetxt(d / f"{seq}.txt", out, delimiter=",", fmt="%.2f")

    (seqmap_dir / f"{bench_split}.txt").write_text("\n".join(seqs) + "\n")
    print(f"seqmap -> {seqmap_dir / (bench_split + '.txt')}")

    # Our own evaluator on exactly the same converted files (per sequence)
    ours = {}
    for trk in args.trackers:
        per_seq = {}
        for seq in seqs:
            gt_p = gt_root / seq / "gt" / "gt.txt"
            pr_p = trk_root / trk / "data" / f"{seq}.txt"
            if not pr_p.exists():
                continue
            r = evaluate(gt_p, pr_p, 0, (1,), 0.5)
            per_seq[seq] = {k: float(r[k]) for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA")}
            per_seq[seq]["IDSW"] = int(r["IDSW"])
        if per_seq:
            keys = ("HOTA", "DetA", "AssA", "IDF1", "MOTA")
            ours[trk] = {
                "per_sequence": per_seq,
                "mean": {k: float(np.mean([v[k] for v in per_seq.values()])) for k in keys},
            }
            ours[trk]["mean"]["IDSW"] = int(sum(v["IDSW"] for v in per_seq.values()))
    (args.out / "ours_metrics.json").write_text(json.dumps(ours, indent=2))
    print(json.dumps({t: {k: round(v, 3) for k, v in o["mean"].items()} for t, o in ours.items()}, indent=2))


if __name__ == "__main__":
    main()
