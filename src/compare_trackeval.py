"""Align our HOTA / CLEAR / Identity implementation with TrackEval.

Reads the TrackEval run on the converted (single-class, no-ignore) data and our
own evaluator's output on exactly the same files, then writes a per-sequence
comparison table.

Usage:
    python src/export_trackeval.py                 # build data + our metrics
    python scripts/run_mot_challenge.py ...        # in TrackEval repo
    python src/compare_trackeval.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "outputs" / "trackeval"
TE_ROOT = BASE / "trackers" / "mot_challenge" / "VisDrone-val"


def main() -> None:
    ours = json.loads((BASE / "ours_metrics.json").read_text())
    rows = []
    for trk, data in ours.items():
        te_path = TE_ROOT / trk / "pedestrian_detailed.csv"
        if not te_path.exists():
            continue
        te = {r["seq"]: r for r in csv.DictReader(open(te_path))}
        for seq, o in data["per_sequence"].items():
            t = te[seq]
            rows.append({
                "tracker": trk,
                "sequence": seq,
                "HOTA_ours": round(o["HOTA"], 3),
                "HOTA_trackeval": round(float(t["HOTA___AUC"]) * 100, 3),
                "IDF1_ours": round(o["IDF1"], 3),
                "IDF1_trackeval": round(float(t["IDF1"]) * 100, 3),
                "IDSW_ours": int(o["IDSW"]),
                "IDSW_trackeval": int(t["IDSW"]),
                "MOTA_ours_pct": round(o["MOTA"], 3),
                "MOTA_trackeval_pct": round(float(t["MOTA"]) * 100, 3),
            })

    out = PROJECT_ROOT / "results" / "trackeval_alignment.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    dh = np.array([abs(r["HOTA_ours"] - r["HOTA_trackeval"]) for r in rows])
    di = np.array([abs(r["IDF1_ours"] - r["IDF1_trackeval"]) for r in rows])
    dm = np.array([abs(r["MOTA_ours_pct"] - r["MOTA_trackeval_pct"]) for r in rows])
    print(f"{len(rows)} sequence-tracker pairs")
    print(f"|ΔHOTA|  mean={dh.mean():.4f} max={dh.max():.4f}")
    print(f"|ΔIDF1|  mean={di.mean():.4f} max={di.max():.4f}")
    print(f"|ΔMOTA|  mean={dm.mean():.4f} max={dm.max():.4f}")
    print(f"IDSW identical: {all(r['IDSW_ours'] == r['IDSW_trackeval'] for r in rows)}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
