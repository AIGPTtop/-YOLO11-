"""Run the full MOT-val evaluation with a patched GMC (downscale / method).

Ultralytics builds `GMC(method=args.gmc_method)` with a fixed downscale of 2,
so the knobs have to be injected before the tracker is constructed.

Usage:
    python src/run_motval_gmc.py --downscale 4 --output outputs/motval_gmc_ds4
    python src/run_motval_gmc.py --method none --output outputs/motval_gmc_none
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--method", type=str, default="", help="override gmc_method, e.g. orb / ecc / none")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--trackers", nargs="+", default=["botsort_gmc"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="0")
    args = ap.parse_args()

    import ultralytics.trackers.bot_sort as bot_sort
    from ultralytics.trackers.utils import gmc as gmc_mod

    ds, method_override = args.downscale, args.method

    class PatchedGMC(gmc_mod.GMC):
        def __init__(self, method: str = "sparseOptFlow", downscale: int = 2) -> None:
            super().__init__(method=(method_override or method), downscale=ds)

    bot_sort.GMC = PatchedGMC

    import run_motval_all

    sys.argv = [
        "run_motval_all.py",
        "--trackers", *args.trackers,
        "--conf", str(args.conf),
        "--device", args.device,
        "--output", args.output,
        "--overwrite",
    ]
    run_motval_all.main()


if __name__ == "__main__":
    main()
