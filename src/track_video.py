# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = PROJECT_ROOT / "weights" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs"

VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO object tracking on a video with ByteTrack."
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT
                / "datasets"
                / "VisDrone2019-MOT-val"
                / "videos"
                / "uav0000305_00000_v.mp4",
        help="Path to a clean input video without ground-truth boxes drawn on it.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the trained best.pt model.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory used by Ultralytics to save tracking results.",
    )

    parser.add_argument(
        "--name",
        default="track_botsort_gmc",  # track_bytetrack
        help="Output run name.",
    )

    parser.add_argument(
        "--tracker",
        default=str(PROJECT_ROOT / "configs" / "botsort_visdrone_gmc.yaml"),  # or "bytetrack.yaml"
        help="Ultralytics tracker config name or path.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="Inference device, for example 0 or cpu.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}"
        )

    if not args.source.is_file():
        raise FileNotFoundError(
            f"Video not found: {args.source}"
        )

    if args.source.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(
            f"Unsupported video format: "
            f"{args.source.suffix or '<no suffix>'}"
        )

    if args.imgsz <= 0:
        raise ValueError(
            "--imgsz must be positive"
        )

    if not 0.0 <= args.conf <= 1.0:
        raise ValueError(
            "--conf must be between 0 and 1"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("YOLO + ByteTrack video tracking")
    print("=" * 70)

    print(f"Model   : {args.model}")
    print(f"Source  : {args.source}")
    print(f"Tracker : {args.tracker}")
    print(f"Output  : {args.output / args.name}")

    model = YOLO(str(args.model))

    results = model.track(
        source=str(args.source),
        tracker=args.tracker,

        imgsz=args.imgsz,
        conf=args.conf,

        device=args.device,

        # VisDrone scenes can contain many objects.
        max_det=1000,

        # Save annotated tracking video.
        save=True,

        # Avoid keeping all video results in memory.
        stream=True,

        project=str(args.output),
        name=args.name,
        exist_ok=True,

        verbose=False,
    )

    frame_count = 0
    tracked_frames = 0

    unique_track_ids: set[int] = set()

    for result in results:
        frame_count += 1

        boxes = result.boxes

        if (
            boxes is not None
            and boxes.is_track
            and boxes.id is not None
        ):
            tracked_frames += 1

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )

            unique_track_ids.update(
                int(track_id)
                for track_id in track_ids
            )

    if frame_count == 0:
        raise RuntimeError(
            "No video frames were processed."
        )

    print()
    print("Finished.")
    print(f"Processed frames : {frame_count}")
    print(f"Tracked frames   : {tracked_frames}")
    print(f"Unique track IDs : {len(unique_track_ids)}")
    print(
        f"Saved under      : "
        f"{args.output / args.name}"
    )


if __name__ == "__main__":
    main()