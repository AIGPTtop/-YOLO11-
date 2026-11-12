from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
from ultralytics import YOLO

from evaluate_mot import evaluate


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRACKERS = {
    "bytetrack": "bytetrack.yaml",

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

    "botsort_reid": str(
        PROJECT_ROOT
        / "configs"
        / "botsort_visdrone_gmc_reid.yaml"
    ),
}

VALID_CLASSES = tuple(
    range(1, 11)
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run full VisDrone2019-MOT-val "
            "tracking evaluation."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "datasets"
            / "VisDrone2019-MOT-val"
        ),
        help=(
            "Dataset root containing "
            "sequences/ and annotations/."
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
            / "motval_full"
        ),
    )

    parser.add_argument(
        "--trackers",
        nargs="+",
        choices=tuple(TRACKERS),
        default=[
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
        "--match-iou",
        type=float,
        default=0.50,
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
        "--limit",
        type=int,
        default=0,
        help=(
            "Only run first N sequences. "
            "0 means all."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-run tracking even if "
            "prediction txt already exists."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:

    sequences_dir = (
        args.dataset
        / "sequences"
    )

    annotations_dir = (
        args.dataset
        / "annotations"
    )

    if not sequences_dir.is_dir():
        raise FileNotFoundError(
            f"Missing sequences directory: "
            f"{sequences_dir}"
        )

    if not annotations_dir.is_dir():
        raise FileNotFoundError(
            f"Missing annotations directory: "
            f"{annotations_dir}"
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

    if not 0.0 <= args.conf <= 1.0:
        raise ValueError(
            "--conf must be in [0, 1]"
        )

    if not 0.0 <= args.iou <= 1.0:
        raise ValueError(
            "--iou must be in [0, 1]"
        )

    if not 0.0 < args.match_iou <= 1.0:
        raise ValueError(
            "--match-iou must be in (0, 1]"
        )

    if args.max_det <= 0:
        raise ValueError(
            "--max-det must be > 0"
        )

    if args.limit < 0:
        raise ValueError(
            "--limit must be >= 0"
        )

    for tracker_name in args.trackers:

        tracker = TRACKERS[
            tracker_name
        ]

        if (
            tracker.endswith(".yaml")
            and tracker not in {
                "bytetrack.yaml",
                "botsort.yaml",
            }
        ):

            if not Path(
                tracker
            ).is_file():

                raise FileNotFoundError(
                    f"Tracker config for "
                    f"{tracker_name!r} "
                    f"not found: "
                    f"{tracker}"
                )


def discover_sequences(
    dataset: Path,
    limit: int = 0,
):

    sequences_dir = (
        dataset
        / "sequences"
    )

    annotations_dir = (
        dataset
        / "annotations"
    )

    sequence_dirs = {
        p.name: p
        for p in sequences_dir.iterdir()
        if p.is_dir()
    }

    gt_files = {
        p.stem: p
        for p in annotations_dir.glob(
            "*.txt"
        )
        if p.is_file()
    }

    missing_gt = sorted(
        set(sequence_dirs)
        - set(gt_files)
    )

    missing_seq = sorted(
        set(gt_files)
        - set(sequence_dirs)
    )

    if missing_gt or missing_seq:

        details = []

        if missing_gt:
            details.append(
                "sequences without GT: "
                f"{missing_gt[:5]}"
            )

        if missing_seq:
            details.append(
                "GT without sequence: "
                f"{missing_seq[:5]}"
            )

        raise RuntimeError(
            "VisDrone MOT-val is incomplete "
            "or names do not match: "
            + "; ".join(details)
        )

    names = sorted(
        sequence_dirs
    )

    if limit:
        names = names[:limit]

    if not names:
        raise RuntimeError(
            "No MOT sequences found."
        )

    return [
        (
            name,
            sequence_dirs[name],
            gt_files[name],
        )
        for name in names
    ]


def list_frames(
    sequence_dir: Path,
) -> list[Path]:

    frames = sorted(
        sequence_dir.glob(
            "*.jpg"
        )
    )

    if not frames:
        frames = sorted(
            sequence_dir.glob(
                "*.png"
            )
        )

    if not frames:
        raise RuntimeError(
            f"No JPG/PNG frames found in "
            f"{sequence_dir}"
        )

    return frames


def write_prediction_rows(
    writer,
    frame_index: int,
    result,
) -> int:

    boxes = result.boxes

    if (
        boxes is None
        or boxes.id is None
        or not boxes.is_track
    ):
        return 0

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

    written = 0

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

        writer.writerow(
            [
                frame_index,

                int(
                    track_id
                ),

                f"{x1:.3f}",

                f"{y1:.3f}",

                f"{width:.3f}",

                f"{height:.3f}",

                f"{float(score):.6f}",

                # YOLO:
                # 0 ~ 9
                #
                # VisDrone:
                # 1 ~ 10
                int(cls0) + 1,

                -1,
                -1,
            ]
        )

        written += 1

    return written


def track_sequence(
    sequence_name: str,
    sequence_dir: Path,
    model_path: Path,
    tracker_config: str,
    pred_path: Path,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    progress_every: int,
) -> int:

    frame_paths = list_frames(
        sequence_dir
    )


    model = YOLO(
        str(
            model_path
        )
    )

    pred_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_count = 0

    with pred_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        for (
            frame_index,
            frame_path,
        ) in enumerate(
            frame_paths,
            start=1,
        ):

            frame = cv2.imread(
                str(
                    frame_path
                )
            )

            if frame is None:
                raise RuntimeError(
                    f"Failed to read frame: "
                    f"{frame_path}"
                )

            results = model.track(
                frame,

                # �����ֶ���֡���룬
                # ������Ҫ���� tracker state��
                persist=True,

                tracker=tracker_config,

                imgsz=imgsz,

                conf=conf,

                iou=iou,

                max_det=max_det,

                device=device,

                verbose=False,
            )

            if not results:
                raise RuntimeError(
                    f"{sequence_name}: "
                    f"no result for frame "
                    f"{frame_index}"
                )

            row_count += (
                write_prediction_rows(
                    writer,
                    frame_index,
                    results[0],
                )
            )

            if (
                progress_every
                and (
                    frame_index
                    % progress_every
                    == 0
                    or frame_index
                    == len(
                        frame_paths
                    )
                )
            ):

                print(
                    f"    {sequence_name}: "
                    f"{frame_index}/"
                    f"{len(frame_paths)} "
                    f"frames, "
                    f"{row_count} rows"
                )

    return len(
        frame_paths
    )


def read_numeric_rows(
    path: Path,
):

    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:

        for (
            line_no,
            line,
        ) in enumerate(
            f,
            start=1,
        ):

            line = (
                line
                .strip()
                .rstrip(",")
            )

            if not line:
                continue

            parts = [
                x.strip()
                for x in line.split(",")
            ]

            if len(parts) < 7:
                raise ValueError(
                    f"{path}:{line_no}: "
                    f"expected >= 7 columns"
                )

            try:

                row = [
                    float(x)
                    for x in parts
                ]

            except ValueError as exc:

                raise ValueError(
                    f"{path}:{line_no}: "
                    f"non-numeric field"
                ) from exc

            row += (
                [-1.0]
                * (
                    10
                    - len(row)
                )
            )

            rows.append(
                row[:10]
            )

    return rows


def remap_sequence_rows(
    rows,
    *,
    frame_offset: int,
    id_map: dict[int, int],
    next_id: int,
):

    output = []

    for original in rows:

        row = list(
            original
        )

        row[0] = (
            int(row[0])
            + frame_offset
        )

        old_id = int(
            row[1]
        )

        if old_id >= 0:

            if old_id not in id_map:

                id_map[
                    old_id
                ] = next_id

                next_id += 1

            row[1] = id_map[
                old_id
            ]

        output.append(
            row
        )

    return (
        output,
        next_id,
    )


def write_numeric_rows(
    path: Path,
    rows,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerows(
            rows
        )


def save_json(
    path: Path,
    data: dict,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():

    args = parse_args()

    validate_args(
        args
    )

    sequences = discover_sequences(
        args.dataset,
        args.limit,
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 72
    )

    print(
        "Full VisDrone2019-MOT-val Evaluation"
    )

    print(
        "=" * 72
    )

    print(
        f"Sequences : "
        f"{len(sequences)}"
    )

    print(
        f"Trackers  : "
        f"{', '.join(args.trackers)}"
    )

    print(
        f"Model     : "
        f"{args.model}"
    )

    print(
        f"imgsz     : "
        f"{args.imgsz}"
    )

    print(
        f"conf/iou  : "
        f"{args.conf}/"
        f"{args.iou}"
    )

    print(
        f"match IoU : "
        f"{args.match_iou}"
    )

    print()

    sequence_info = []

    for (
        sequence_name,
        sequence_dir,
        gt_path,
    ) in sequences:

        frame_count = len(
            list_frames(
                sequence_dir
            )
        )

        sequence_info.append(
            {
                "name": (
                    sequence_name
                ),

                "sequence_dir": (
                    sequence_dir
                ),

                "gt_path": (
                    gt_path
                ),

                "frames": (
                    frame_count
                ),

                "gt_rows": (
                    read_numeric_rows(
                        gt_path
                    )
                ),
            }
        )

    summary_rows = []

    global_metrics_by_tracker = {}

    for tracker_name in args.trackers:

        tracker_config = (
            TRACKERS[
                tracker_name
            ]
        )

        tracker_dir = (
            args.output
            / tracker_name
        )

        tracker_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "-" * 72
        )

        print(
            f"Tracker: "
            f"{tracker_name}"
        )

        print(
            f"Config : "
            f"{tracker_config}"
        )

        combined_gt = []

        combined_pred = []

        frame_offset = 0

        next_gt_id = 0

        next_pred_id = 0

        for (
            seq_idx,
            info,
        ) in enumerate(
            sequence_info,
            start=1,
        ):

            name = info[
                "name"
            ]

            sequence_dir = info[
                "sequence_dir"
            ]

            gt_path = info[
                "gt_path"
            ]

            frame_count = info[
                "frames"
            ]

            pred_path = (
                tracker_dir
                / f"{name}.txt"
            )

            metrics_path = (
                tracker_dir
                / f"{name}_metrics.json"
            )

            print(
                f"[{seq_idx}/"
                f"{len(sequence_info)}] "
                f"{name}"
            )

            if (
                args.overwrite
                or not pred_path.is_file()
            ):

                processed = (
                    track_sequence(
                        name,

                        sequence_dir,

                        args.model,

                        tracker_config,

                        pred_path,

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

                        progress_every=(
                            args.progress_every
                        ),
                    )
                )

                if (
                    processed
                    != frame_count
                ):

                    raise RuntimeError(
                        f"{name}: "
                        f"processed "
                        f"{processed} "
                        f"frames but expected "
                        f"{frame_count}"
                    )

            else:

                print(
                    "    reuse existing "
                    f"prediction: "
                    f"{pred_path}"
                )

            metrics = evaluate(
                gt_path=gt_path,

                pred_path=pred_path,

                num_frames=(
                    frame_count
                ),

                classes=(
                    VALID_CLASSES
                ),

                match_iou=(
                    args.match_iou
                ),
            )

            save_json(
                metrics_path,
                metrics,
            )

            row = {
                "sequence": name,

                "tracker": (
                    tracker_name
                ),

                **metrics,
            }

            summary_rows.append(
                row
            )

            print(
                f"    HOTA="
                f"{metrics['HOTA']:.2f}  "

                f"IDF1="
                f"{metrics['IDF1']:.2f}  "

                f"MOTA="
                f"{metrics['MOTA']:.2f}  "

                f"IDSW="
                f"{metrics['IDSW']}  "

                f"Frag="
                f"{metrics['Frag']}"
            )

            # -----------------------------
            # Full-dataset aggregation
            # -----------------------------
            #
            # ÿ�� sequence��
            #
            # frame �������ƫ�ƣ�
            # ID ����ӳ���ȫ��Ψһ ID��
            #
            # ���ⲻͬ sequence �ģ�
            #
            # frame=1
            # ID=1
            #
            # ������Ϊͬһ���켣��

            gt_map = {}

            pred_map = {}

            (
                gt_rows,
                next_gt_id,
            ) = remap_sequence_rows(
                info[
                    "gt_rows"
                ],

                frame_offset=(
                    frame_offset
                ),

                id_map=(
                    gt_map
                ),

                next_id=(
                    next_gt_id
                ),
            )

            (
                pred_rows,
                next_pred_id,
            ) = remap_sequence_rows(
                read_numeric_rows(
                    pred_path
                ),

                frame_offset=(
                    frame_offset
                ),

                id_map=(
                    pred_map
                ),

                next_id=(
                    next_pred_id
                ),
            )

            combined_gt.extend(
                gt_rows
            )

            combined_pred.extend(
                pred_rows
            )

            frame_offset += (
                frame_count
            )

        combined_gt_path = (
            tracker_dir
            / "_combined_gt.txt"
        )

        combined_pred_path = (
            tracker_dir
            / "_combined_pred.txt"
        )

        write_numeric_rows(
            combined_gt_path,
            combined_gt,
        )

        write_numeric_rows(
            combined_pred_path,
            combined_pred,
        )

        global_metrics = evaluate(
            gt_path=(
                combined_gt_path
            ),

            pred_path=(
                combined_pred_path
            ),

            num_frames=(
                frame_offset
            ),

            classes=(
                VALID_CLASSES
            ),

            match_iou=(
                args.match_iou
            ),
        )

        global_metrics_by_tracker[
            tracker_name
        ] = global_metrics

        save_json(
            tracker_dir
            / "GLOBAL_metrics.json",

            global_metrics,
        )

        print()

        print(
            f"GLOBAL "
            f"{tracker_name}:"
        )

        print(
            f"  HOTA="
            f"{global_metrics['HOTA']:.2f}  "

            f"DetA="
            f"{global_metrics['DetA']:.2f}  "

            f"AssA="
            f"{global_metrics['AssA']:.2f}  "

            f"IDF1="
            f"{global_metrics['IDF1']:.2f}  "

            f"MOTA="
            f"{global_metrics['MOTA']:.2f}  "

            f"IDSW="
            f"{global_metrics['IDSW']}  "

            f"Frag="
            f"{global_metrics['Frag']}"
        )

        print()

    # --------------------------------------
    # Per-sequence CSV
    # --------------------------------------

    summary_csv = (
        args.output
        / "per_sequence_metrics.csv"
    )

    if summary_rows:

        fieldnames = list(
            summary_rows[
                0
            ].keys()
        )

        with summary_csv.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                summary_rows
            )

    # --------------------------------------
    # Global summary
    # --------------------------------------

    global_json = (
        args.output
        / "GLOBAL_summary.json"
    )

    save_json(
        global_json,
        global_metrics_by_tracker,
    )

    global_csv = (
        args.output
        / "GLOBAL_summary.csv"
    )

    global_rows = [
        {
            "tracker": (
                tracker_name
            ),

            **metrics,
        }
        for (
            tracker_name,
            metrics,
        )
        in global_metrics_by_tracker.items()
    ]

    if global_rows:

        with global_csv.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    global_rows[
                        0
                    ].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                global_rows
            )

    print(
        "=" * 72
    )

    print(
        "Finished"
    )

    print(
        f"Per-sequence CSV : "
        f"{summary_csv}"
    )

    print(
        f"Global CSV       : "
        f"{global_csv}"
    )

    print(
        f"Global JSON      : "
        f"{global_json}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()