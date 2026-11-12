from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO11 MOT tracking + VisDrone-format txt export"
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=(
            PROJECT_ROOT
            / "datasets"
            / "VisDrone2019-MOT-val"
            / "videos"
            / "uav0000305_00000_v.mp4"
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "weights" / "best.pt",
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default=str(
            PROJECT_ROOT
            / "configs"
            / "botsort_visdrone_gmc.yaml"
        ),
        help=(
            "Custom tracker yaml path, or Ultralytics built-in "
            "bytetrack.yaml / botsort.yaml"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "track_botsort_gmc",
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
        "--device",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=1000,
    )

    return parser.parse_args()


def validate(args):
    if not args.source.is_file():
        raise FileNotFoundError(
            f"Video not found: {args.source}"
        )

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}"
        )

    tracker_path = Path(args.tracker)

    if (
        not tracker_path.is_file()
        and args.tracker
        not in {
            "bytetrack.yaml",
            "botsort.yaml",
        }
    ):
        raise FileNotFoundError(
            f"Tracker config not found: {args.tracker}"
        )

    if not 0 <= args.conf <= 1:
        raise ValueError(
            "--conf must be in [0, 1]"
        )

    if not 0 <= args.iou <= 1:
        raise ValueError(
            "--iou must be in [0, 1]"
        )


def main():
    args = parse_args()
    validate(args)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 例如：
    # uav0000305_00000_v.mp4
    # ->
    # uav0000305_00000_v
    seq_name = args.source.stem

    mot_path = (
        args.output
        / f"{seq_name}.txt"
    )

    print("=" * 60)
    print("YOLO11 MOT Tracking")
    print("=" * 60)

    print(f"Source  : {args.source}")
    print(f"Model   : {args.model}")
    print(f"Tracker : {args.tracker}")
    print(f"Output  : {args.output}")

    model = YOLO(
        str(args.model)
    )

    results = model.track(
        source=str(args.source),

        tracker=args.tracker,

        imgsz=args.imgsz,

        conf=args.conf,

        iou=args.iou,

        device=args.device,

        max_det=args.max_det,

        # 保存带框视频
        save=True,

        # 视频逐帧返回结果
        stream=True,

        # 保持 Track ID
        persist=True,

        project=str(args.output),

        name="video",

        exist_ok=True,

        verbose=False,
    )

    frame_count = 0
    row_count = 0

    unique_ids = set()

    with mot_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        for frame_count, result in enumerate(
            results,
            start=1,
        ):

            boxes = result.boxes

            # 当前帧没有有效跟踪目标
            if (
                boxes is None
                or boxes.id is None
                or not boxes.is_track
            ):
                continue

            # xyxy:
            # [x1, y1, x2, y2]
            xyxy = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            track_ids = (
                boxes.id
                .detach()
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            classes = (
                boxes.cls
                .detach()
                .cpu()
                .numpy()
                .astype(int)
            )

            for (
                box,
                track_id,
                score,
                cls0,
            ) in zip(
                xyxy,
                track_ids,
                confidences,
                classes,
            ):

                x1, y1, x2, y2 = map(
                    float,
                    box,
                )

                width = max(
                    0.0,
                    x2 - x1,
                )

                height = max(
                    0.0,
                    y2 - y1,
                )

                # VisDrone MOT 格式：
                #
                # frame,
                # id,
                # x,
                # y,
                # width,
                # height,
                # score,
                # class,
                # truncation,
                # occlusion
                #
                # 你的 YOLO 模型：
                # class = 0 ~ 9
                #
                # VisDrone：
                # class = 1 ~ 10
                #
                # 所以这里 +1

                writer.writerow(
                    [
                        frame_count,

                        int(track_id),

                        f"{x1:.3f}",

                        f"{y1:.3f}",

                        f"{width:.3f}",

                        f"{height:.3f}",

                        f"{float(score):.6f}",

                        int(cls0) + 1,

                        -1,

                        -1,
                    ]
                )

                row_count += 1

                unique_ids.add(
                    int(track_id)
                )

    if frame_count == 0:
        raise RuntimeError(
            "No frames were processed"
        )

    print()
    print("=" * 60)
    print("Tracking finished")
    print("=" * 60)

    print(
        f"Frames           : "
        f"{frame_count}"
    )

    print(
        f"MOT rows         : "
        f"{row_count}"
    )

    print(
        f"Unique track IDs : "
        f"{len(unique_ids)}"
    )

    print(
        f"MOT txt          : "
        f"{mot_path}"
    )

    print(
        f"Video output     : "
        f"{args.output / 'video'}"
    )


if __name__ == "__main__":
    main()