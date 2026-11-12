from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics

from ultralytics import YOLO


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


MODES = {
    "detect": None,

    "bytetrack": (
        "bytetrack.yaml"
    ),

    "botsort_no_gmc": str(
        PROJECT_ROOT
        / "configs"
        / "botsort_visdrone_no_gmc.yaml"
    ),

    "botsort_gmc": str(
        PROJECT_ROOT
        / "configs"
        / "botsort_visdrone_gmc.yaml"
    ),
}


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Formal YOLO11 detection/"
            "tracking FPS and latency benchmark."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,

        default=(
            PROJECT_ROOT
            / "datasets"
            / "VisDrone2019-MOT-val"
            / "sequences"
            / "uav0000305_00000_v"
        ),

        help=(
            "One VisDrone sequence "
            "directory or one video file."
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,

        default=(
            PROJECT_ROOT
            / "weights"
            / "best.pt"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,

        default=(
            PROJECT_ROOT
            / "outputs"
            / "benchmark"
        ),
    )

    parser.add_argument(
        "--modes",
        nargs="+",

        choices=tuple(
            MODES
        ),

        default=[
            "detect",
            "bytetrack",
            "botsort_no_gmc",
            "botsort_gmc",
        ],
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,

        help=(
            "Warm-up frames excluded "
            "from statistics."
        ),
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=150,

        help=(
            "Measured frames after "
            "warm-up."
        ),
    )

    parser.add_argument(
        "--half",
        action="store_true",

        help=(
            "Use FP16 for PyTorch CUDA "
            "benchmark. Default FP32."
        ),
    )

    return parser.parse_args()


def validate_args(
    args,
):

    if not args.source.exists():
        raise FileNotFoundError(
            f"Missing source: "
            f"{args.source}"
        )

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Missing model: "
            f"{args.model}"
        )

    if args.imgsz <= 0:
        raise ValueError(
            "--imgsz must be > 0"
        )

    if not (
        0.0
        <= args.conf
        <= 1.0
    ):
        raise ValueError(
            "--conf must be in [0, 1]"
        )

    if not (
        0.0
        <= args.iou
        <= 1.0
    ):
        raise ValueError(
            "--iou must be in [0, 1]"
        )

    if args.max_det <= 0:
        raise ValueError(
            "--max-det must be > 0"
        )

    if args.warmup < 0:
        raise ValueError(
            "--warmup must be >= 0"
        )

    if args.frames <= 0:
        raise ValueError(
            "--frames must be > 0"
        )

    for mode in args.modes:

        tracker = MODES[
            mode
        ]

        if (
            tracker
            and tracker not in {
                "bytetrack.yaml",
                "botsort.yaml",
            }
        ):

            if not Path(
                tracker
            ).is_file():

                raise FileNotFoundError(
                    f"Tracker config "
                    f"for {mode!r} "
                    f"not found: "
                    f"{tracker}"
                )

    if (
        args.half
        and args.device.lower()
        == "cpu"
    ):

        raise ValueError(
            "--half is intended "
            "for CUDA benchmarking."
        )


def cuda_index(
    device: str,
):

    device = (
        str(device)
        .strip()
        .lower()
    )

    if (
        device == "cpu"
        or not torch.cuda.is_available()
    ):
        return None

    if device.startswith(
        "cuda:"
    ):

        suffix = device.split(
            ":",
            1,
        )[1]

        if suffix.isdigit():
            return int(
                suffix
            )

        return 0

    if device.isdigit():
        return int(
            device
        )

    return 0


def sync_cuda(
    index,
):

    if index is not None:

        torch.cuda.synchronize(
            index
        )


def iter_source_frames(
    source: Path,
):

    # ----------------------------------
    # VisDrone sequence folder
    # ----------------------------------

    if source.is_dir():

        frame_paths = sorted(
            source.glob(
                "*.jpg"
            )
        )

        if not frame_paths:

            frame_paths = sorted(
                source.glob(
                    "*.png"
                )
            )

        if not frame_paths:

            raise RuntimeError(
                f"No JPG/PNG frames "
                f"found in {source}"
            )

        for frame_path in frame_paths:

            t0 = time.perf_counter()

            frame = cv2.imread(
                str(
                    frame_path
                )
            )

            io_ms = (
                time.perf_counter()
                - t0
            ) * 1000.0

            if frame is None:
                raise RuntimeError(
                    f"Failed to read frame: "
                    f"{frame_path}"
                )

            yield (
                frame,
                io_ms,
            )

        return

    # ----------------------------------
    # MP4 / video file
    # ----------------------------------

    cap = cv2.VideoCapture(
        str(
            source
        )
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{source}"
        )

    try:

        while True:

            t0 = (
                time.perf_counter()
            )

            ok, frame = (
                cap.read()
            )

            io_ms = (
                time.perf_counter()
                - t0
            ) * 1000.0

            if not ok:
                break

            yield (
                frame,
                io_ms,
            )

    finally:

        cap.release()


def percentile(
    values,
    q,
):

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            q,
        )
    )


def run_mode(
    args,
    mode: str,
):

    tracker = MODES[
        mode
    ]



    model = YOLO(
        str(
            args.model
        )
    )

    gpu_index = cuda_index(
        args.device
    )

    if gpu_index is not None:

        torch.cuda.set_device(
            gpu_index
        )

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats(
            gpu_index
        )

    compute_ms = []

    io_ms_list = []

    warmup_seen = 0

    measured = 0

    for (
        frame,
        io_ms,
    ) in iter_source_frames(
        args.source
    ):

        is_warmup = (
            warmup_seen
            < args.warmup
        )



        sync_cuda(
            gpu_index
        )

        t0 = (
            time.perf_counter()
        )

        if mode == "detect":

            model.predict(
                frame,

                imgsz=(
                    args.imgsz
                ),

                conf=(
                    args.conf
                ),

                iou=(
                    args.iou
                ),

                max_det=(
                    args.max_det
                ),

                device=(
                    args.device
                ),

                half=(
                    args.half
                ),

                verbose=False,
            )

        else:

            model.track(
                frame,

                persist=True,

                tracker=(
                    tracker
                ),

                imgsz=(
                    args.imgsz
                ),

                conf=(
                    args.conf
                ),

                iou=(
                    args.iou
                ),

                max_det=(
                    args.max_det
                ),

                device=(
                    args.device
                ),

                half=(
                    args.half
                ),

                verbose=False,
            )

        sync_cuda(
            gpu_index
        )

        elapsed_ms = (
            time.perf_counter()
            - t0
        ) * 1000.0

        # ------------------------------
        # Warmup
        # ------------------------------

        if is_warmup:

            warmup_seen += 1

            continue

        # ------------------------------
        # Formal measurement
        # ------------------------------

        compute_ms.append(
            elapsed_ms
        )

        io_ms_list.append(
            io_ms
        )

        measured += 1

        if measured >= args.frames:
            break

    if measured == 0:

        raise RuntimeError(
            f"{mode}: no measured frames. "
            f"Source ended during warm-up "
            f"({args.warmup} requested)."
        )

    if measured < args.frames:

        print(
            f"WARNING: "
            f"{mode}: requested "
            f"{args.frames} frames "
            f"but source provided "
            f"only {measured}."
        )

    compute = np.asarray(
        compute_ms,
        dtype=np.float64,
    )

    io_arr = np.asarray(
        io_ms_list,
        dtype=np.float64,
    )

    e2e = (
        compute
        + io_arr
    )

    peak_gpu_mb = 0.0

    if gpu_index is not None:

        peak_gpu_mb = (
            torch.cuda
            .max_memory_allocated(
                gpu_index
            )
            / (
                1024.0 ** 2
            )
        )

    return {
        "mode": mode,

        "tracker": (
            tracker
            or ""
        ),

        "precision": (
            "FP16"
            if args.half
            else "FP32"
        ),

        "warmup_frames": (
            warmup_seen
        ),

        "measured_frames": (
            measured
        ),

        "compute_mean_ms": float(
            compute.mean()
        ),

        "compute_p50_ms": (
            percentile(
                compute_ms,
                50,
            )
        ),

        "compute_p95_ms": (
            percentile(
                compute_ms,
                95,
            )
        ),

        "compute_p99_ms": (
            percentile(
                compute_ms,
                99,
            )
        ),

        "compute_fps": float(
            1000.0
            / compute.mean()
        ),

        "io_mean_ms": float(
            io_arr.mean()
        ),

        "e2e_mean_ms": float(
            e2e.mean()
        ),

        "e2e_p95_ms": float(
            np.percentile(
                e2e,
                95,
            )
        ),

        "e2e_fps": float(
            1000.0
            / e2e.mean()
        ),

        "peak_gpu_mb": float(
            peak_gpu_mb
        ),
    }


def environment_info(
    args,
):

    gpu_index = cuda_index(
        args.device
    )

    gpu_name = "CPU"

    if gpu_index is not None:

        gpu_name = (
            torch.cuda
            .get_device_name(
                gpu_index
            )
        )

    return {
        "python": (
            platform.python_version()
        ),

        "torch": (
            torch.__version__
        ),

        "ultralytics": (
            getattr(
                ultralytics,
                "__version__",
                "unknown",
            )
        ),

        "cuda_runtime": (
            torch.version.cuda
        ),

        "device_argument": (
            args.device
        ),

        "gpu": (
            gpu_name
        ),

        "model": str(
            args.model
        ),

        "source": str(
            args.source
        ),

        "imgsz": (
            args.imgsz
        ),

        "conf": (
            args.conf
        ),

        "iou": (
            args.iou
        ),

        "max_det": (
            args.max_det
        ),

        "batch": 1,

        "precision": (
            "FP16"
            if args.half
            else "FP32"
        ),

        "warmup": (
            args.warmup
        ),

        "requested_measured_frames": (
            args.frames
        ),
    }


def main():

    args = parse_args()

    validate_args(
        args
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = environment_info(
        args
    )

    print(
        "=" * 76
    )

    print(
        "YOLO11 Runtime Benchmark"
    )

    print(
        "=" * 76
    )

    for (
        key,
        value,
    ) in env.items():

        print(
            f"{key:26s}: "
            f"{value}"
        )

    print()

    rows = []

    for mode in args.modes:

        print(
            f"Benchmarking "
            f"{mode} ..."
        )

        result = run_mode(
            args,
            mode,
        )

        rows.append(
            result
        )

        print(
            f"  compute mean="
            f"{result['compute_mean_ms']:.2f} ms, "

            f"P50="
            f"{result['compute_p50_ms']:.2f} ms, "

            f"P95="
            f"{result['compute_p95_ms']:.2f} ms, "

            f"FPS="
            f"{result['compute_fps']:.2f}, "

            f"E2E FPS="
            f"{result['e2e_fps']:.2f}"
        )

    # ----------------------------------
    # CSV
    # ----------------------------------

    csv_path = (
        args.output
        / "runtime_benchmark.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ----------------------------------
    # JSON
    # ----------------------------------

    json_path = (
        args.output
        / "runtime_benchmark.json"
    )

    json_path.write_text(
        json.dumps(
            {
                "environment": (
                    env
                ),

                "results": (
                    rows
                ),
            },

            ensure_ascii=False,

            indent=2,
        ),

        encoding="utf-8",
    )

    print()

    print(
        "=" * 76
    )

    print(
        f"CSV  : "
        f"{csv_path}"
    )

    print(
        f"JSON : "
        f"{json_path}"
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":
    main()