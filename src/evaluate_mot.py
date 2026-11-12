from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------
# VisDrone category definition
#
# 0  : ignored regions
# 1  : pedestrian
# 2  : people
# 3  : bicycle
# 4  : car
# 5  : van
# 6  : truck
# 7  : tricycle
# 8  : awning-tricycle
# 9  : bus
# 10 : motor
# 11 : others
# ---------------------------------------------------------

VALID_CLASSES = tuple(
    range(1, 11)
)

IGNORE_CLASSES = {
    0,
    11,
}


# HOTA evaluates:
#
# IoU = 0.05
# ...
# IoU = 0.95
#
# Then averages all thresholds.

ALPHAS = np.arange(
    0.05,
    0.99,
    0.05,
    dtype=np.float64,
)

EPS = np.finfo(
    np.float64
).eps


# =========================================================
# 1. Read MOT / VisDrone txt
# =========================================================

def read_rows(
    path: Path,
) -> List[List[float]]:

    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:

        for line_no, line in enumerate(
            f,
            1,
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
                    f"non-numeric value"
                ) from exc

            if int(row[0]) < 1:
                raise ValueError(
                    f"{path}:{line_no}: "
                    f"frame id must start from 1"
                )

            if (
                row[4] < 0
                or row[5] < 0
            ):
                raise ValueError(
                    f"{path}:{line_no}: "
                    f"width/height must be >= 0"
                )

            # 如果 prediction 只有：
            #
            # frame,id,x,y,w,h,score
            #
            # 则自动补到10列。
            #
            # class=-1 表示 class-agnostic。

            row += (
                [-1.0]
                * (10 - len(row))
            )

            rows.append(
                row[:10]
            )

    return rows


# =========================================================
# 2. Group by frame
# =========================================================

def group_by_frame(
    rows: Sequence[
        Sequence[float]
    ],
) -> Dict[
    int,
    List[List[float]],
]:

    output = {}

    for row in rows:

        frame = int(
            row[0]
        )

        output.setdefault(
            frame,
            [],
        ).append(
            list(row)
        )

    return output


# =========================================================
# 3. VisDrone ignored regions
# =========================================================

def center_in_ignore(
    row: Sequence[float],
    ignore_regions: Sequence[
        Sequence[float]
    ],
) -> bool:

    # bbox:
    # x, y, width, height

    cx = (
        row[2]
        + row[4] / 2.0
    )

    cy = (
        row[3]
        + row[5] / 2.0
    )

    for region in ignore_regions:

        x1 = region[2]
        y1 = region[3]

        x2 = (
            region[2]
            + region[4]
        )

        y2 = (
            region[3]
            + region[5]
        )

        if (
            x1 < cx < x2
            and y1 < cy < y2
        ):
            return True

    return False


def filter_visdrone(
    gt_rows: Sequence[
        Sequence[float]
    ],
    pred_rows: Sequence[
        Sequence[float]
    ],
    classes: Sequence[int],
) -> Tuple[
    List[List[float]],
    List[List[float]],
]:

    classes = set(
        classes
    )

    gt_by_frame = group_by_frame(
        gt_rows
    )

    pred_by_frame = group_by_frame(
        pred_rows
    )

    filtered_gt = []
    filtered_pred = []

    all_frames = sorted(
        set(gt_by_frame)
        | set(pred_by_frame)
    )

    for frame in all_frames:

        gt_frame = gt_by_frame.get(
            frame,
            [],
        )

        pred_frame = pred_by_frame.get(
            frame,
            [],
        )

        # VisDrone:
        #
        # class 0  -> ignore region
        # class 11 -> others
        #
        # 不参与正常 MOT 指标。

        ignore_regions = [
            row
            for row in gt_frame
            if int(row[7])
            in IGNORE_CLASSES
        ]

        # -------------------------
        # Ground Truth filtering
        # -------------------------

        for row in gt_frame:

            score = row[6]
            cls = int(
                row[7]
            )

            if score <= 0:
                continue

            if cls not in classes:
                continue

            if center_in_ignore(
                row,
                ignore_regions,
            ):
                continue

            filtered_gt.append(
                row
            )

        # -------------------------
        # Prediction filtering
        # -------------------------

        for row in pred_frame:

            score = row[6]
            cls = int(
                row[7]
            )

            if score <= 0:
                continue

            # -1 表示 prediction
            # 没有类别信息，
            # 按 class-agnostic 处理。

            if (
                cls != -1
                and cls not in classes
            ):
                continue

            if center_in_ignore(
                row,
                ignore_regions,
            ):
                continue

            filtered_pred.append(
                row
            )

    return (
        filtered_gt,
        filtered_pred,
    )


# =========================================================
# 4. Bounding-box IoU
# =========================================================

def iou_xywh(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:

    boxes_a = np.asarray(
        boxes_a,
        dtype=np.float64,
    ).reshape(
        -1,
        4,
    )

    boxes_b = np.asarray(
        boxes_b,
        dtype=np.float64,
    ).reshape(
        -1,
        4,
    )

    if (
        len(boxes_a) == 0
        or len(boxes_b) == 0
    ):

        return np.zeros(
            (
                len(boxes_a),
                len(boxes_b),
            ),
            dtype=np.float64,
        )

    # -------------------------
    # A
    # -------------------------

    ax1 = boxes_a[
        :,
        0,
    ][:, None]

    ay1 = boxes_a[
        :,
        1,
    ][:, None]

    ax2 = (
        boxes_a[:, 0]
        + boxes_a[:, 2]
    )[:, None]

    ay2 = (
        boxes_a[:, 1]
        + boxes_a[:, 3]
    )[:, None]

    # -------------------------
    # B
    # -------------------------

    bx1 = boxes_b[
        :,
        0,
    ][None, :]

    by1 = boxes_b[
        :,
        1,
    ][None, :]

    bx2 = (
        boxes_b[:, 0]
        + boxes_b[:, 2]
    )[None, :]

    by2 = (
        boxes_b[:, 1]
        + boxes_b[:, 3]
    )[None, :]

    # -------------------------
    # Intersection
    # -------------------------

    intersection_width = np.maximum(
        0.0,
        np.minimum(
            ax2,
            bx2,
        )
        - np.maximum(
            ax1,
            bx1,
        ),
    )

    intersection_height = np.maximum(
        0.0,
        np.minimum(
            ay2,
            by2,
        )
        - np.maximum(
            ay1,
            by1,
        ),
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    # -------------------------
    # Union
    # -------------------------

    area_a = np.maximum(
        0.0,
        (
            boxes_a[:, 2]
            * boxes_a[:, 3]
        ),
    )[:, None]

    area_b = np.maximum(
        0.0,
        (
            boxes_b[:, 2]
            * boxes_b[:, 3]
        ),
    )[None, :]

    union = (
        area_a
        + area_b
        - intersection
    )

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(
            intersection
        ),
        where=union > EPS,
    )


# =========================================================
# 5. Remap Track IDs
# =========================================================

def remap_ids(
    ids_by_frame: List[
        np.ndarray
    ],
) -> Tuple[
    List[np.ndarray],
    int,
]:

    non_empty = [
        ids
        for ids in ids_by_frame
        if len(ids)
    ]

    if not non_empty:
        return (
            ids_by_frame,
            0,
        )

    unique_ids = np.unique(
        np.concatenate(
            non_empty
        ).astype(
            np.int64
        )
    )

    if np.any(
        unique_ids < 0
    ):
        raise ValueError(
            "Valid GT/prediction "
            "track IDs must be non-negative"
        )

    mapping = {
        int(old_id): new_id
        for new_id, old_id
        in enumerate(
            unique_ids.tolist()
        )
    }

    remapped = []

    for ids in ids_by_frame:

        if len(ids):

            new_ids = np.asarray(
                [
                    mapping[int(x)]
                    for x in ids
                ],
                dtype=np.int64,
            )

        else:

            new_ids = np.empty(
                0,
                dtype=np.int64,
            )

        remapped.append(
            new_ids
        )

    return (
        remapped,
        len(unique_ids),
    )


# =========================================================
# 6. Build MOT evaluation data
# =========================================================

def build_data(
    gt_rows,
    pred_rows,
    num_frames: int,
    classes: Sequence[int],
):

    (
        gt_rows,
        pred_rows,
    ) = filter_visdrone(
        gt_rows,
        pred_rows,
        classes,
    )

    gt_by_frame = group_by_frame(
        gt_rows
    )

    pred_by_frame = group_by_frame(
        pred_rows
    )

    max_seen = max(
        [0]
        + [
            int(row[0])
            for row in gt_rows
        ]
        + [
            int(row[0])
            for row in pred_rows
        ]
    )

    if num_frames < max_seen:
        raise ValueError(
            f"num_frames={num_frames}, "
            f"but data contains frame "
            f"{max_seen}"
        )

    gt_ids = []
    tracker_ids = []

    gt_boxes = []
    tracker_boxes = []

    for frame in range(
        1,
        num_frames + 1,
    ):

        gt_frame = (
            gt_by_frame.get(
                frame,
                [],
            )
        )

        pred_frame = (
            pred_by_frame.get(
                frame,
                [],
            )
        )

        current_gt_ids = np.asarray(
            [
                int(row[1])
                for row in gt_frame
            ],
            dtype=np.int64,
        )

        current_tracker_ids = np.asarray(
            [
                int(row[1])
                for row in pred_frame
            ],
            dtype=np.int64,
        )

        # 同一帧不能出现两个相同 GT ID

        if (
            len(
                np.unique(
                    current_gt_ids
                )
            )
            != len(
                current_gt_ids
            )
        ):
            raise ValueError(
                f"Duplicate GT ID "
                f"in frame {frame}"
            )

        # 同一帧不能出现两个相同 Tracker ID

        if (
            len(
                np.unique(
                    current_tracker_ids
                )
            )
            != len(
                current_tracker_ids
            )
        ):
            raise ValueError(
                f"Duplicate prediction ID "
                f"in frame {frame}"
            )

        gt_ids.append(
            current_gt_ids
        )

        tracker_ids.append(
            current_tracker_ids
        )

        current_gt_boxes = np.asarray(
            [
                row[2:6]
                for row in gt_frame
            ],
            dtype=np.float64,
        ).reshape(
            -1,
            4,
        )

        current_tracker_boxes = np.asarray(
            [
                row[2:6]
                for row in pred_frame
            ],
            dtype=np.float64,
        ).reshape(
            -1,
            4,
        )

        gt_boxes.append(
            current_gt_boxes
        )

        tracker_boxes.append(
            current_tracker_boxes
        )

    (
        gt_ids,
        num_gt_ids,
    ) = remap_ids(
        gt_ids
    )

    (
        tracker_ids,
        num_tracker_ids,
    ) = remap_ids(
        tracker_ids
    )

    similarities = [
        iou_xywh(
            gt_boxes_frame,
            tracker_boxes_frame,
        )
        for (
            gt_boxes_frame,
            tracker_boxes_frame,
        )
        in zip(
            gt_boxes,
            tracker_boxes,
        )
    ]

    return {
        "gt_ids": gt_ids,

        "tr_ids": tracker_ids,

        "sims": similarities,

        "n_gt_ids": num_gt_ids,

        "n_tr_ids": num_tracker_ids,

        "n_gt_dets": sum(
            len(x)
            for x in gt_ids
        ),

        "n_tr_dets": sum(
            len(x)
            for x in tracker_ids
        ),

        "num_frames": num_frames,
    }


# =========================================================
# 7. HOTA
# =========================================================

def finish_hota(
    true_positive,
    false_negative,
    false_positive,
    association_accuracy,
    localization_accuracy,
):

    detection_accuracy = (
        true_positive
        / np.maximum(
            1.0,
            (
                true_positive
                + false_negative
                + false_positive
            ),
        )
    )

    hota = np.sqrt(
        detection_accuracy
        * association_accuracy
    )

    return {
        "HOTA": hota,

        "DetA": (
            detection_accuracy
        ),

        "AssA": (
            association_accuracy
        ),

        "LocA": (
            localization_accuracy
        ),

        "HOTA_TP": (
            true_positive
        ),

        "HOTA_FN": (
            false_negative
        ),

        "HOTA_FP": (
            false_positive
        ),
    }


def eval_hota(
    data,
):

    num_thresholds = len(
        ALPHAS
    )

    true_positive = np.zeros(
        num_thresholds
    )

    false_negative = np.zeros(
        num_thresholds
    )

    false_positive = np.zeros(
        num_thresholds
    )

    association_accuracy = np.zeros(
        num_thresholds
    )

    localization_accuracy = np.zeros(
        num_thresholds
    )

    # -----------------------------------------------------
    # No GT
    # -----------------------------------------------------

    if data["n_gt_dets"] == 0:

        false_positive[:] = (
            data["n_tr_dets"]
        )

        return finish_hota(
            true_positive,
            false_negative,
            false_positive,
            association_accuracy,
            np.ones(
                num_thresholds
            ),
        )

    # -----------------------------------------------------
    # No tracker results
    # -----------------------------------------------------

    if data["n_tr_dets"] == 0:

        false_negative[:] = (
            data["n_gt_dets"]
        )

        return finish_hota(
            true_positive,
            false_negative,
            false_positive,
            association_accuracy,
            np.ones(
                num_thresholds
            ),
        )

    # -----------------------------------------------------
    # Global association statistics
    # -----------------------------------------------------

    potential_matches = np.zeros(
        (
            data["n_gt_ids"],
            data["n_tr_ids"],
        )
    )

    gt_id_count = np.zeros(
        (
            data["n_gt_ids"],
            1,
        )
    )

    tracker_id_count = np.zeros(
        (
            1,
            data["n_tr_ids"],
        )
    )

    # -----------------------------------------------------
    # First pass
    # -----------------------------------------------------

    for (
        gt_ids_frame,
        tracker_ids_frame,
        similarity,
    ) in zip(
        data["gt_ids"],
        data["tr_ids"],
        data["sims"],
    ):

        if (
            len(gt_ids_frame)
            and len(tracker_ids_frame)
        ):

            denominator = (
                similarity.sum(
                    axis=0
                )[None, :]
                + similarity.sum(
                    axis=1
                )[:, None]
                - similarity
            )

            similarity_iou = np.divide(
                similarity,
                denominator,
                out=np.zeros_like(
                    similarity
                ),
                where=(
                    denominator
                    > EPS
                ),
            )

            potential_matches[
                gt_ids_frame[:, None],
                tracker_ids_frame[None, :],
            ] += similarity_iou

        if len(
            gt_ids_frame
        ):

            gt_id_count[
                gt_ids_frame
            ] += 1

        if len(
            tracker_ids_frame
        ):

            tracker_id_count[
                0,
                tracker_ids_frame,
            ] += 1

    # -----------------------------------------------------
    # Global alignment
    # -----------------------------------------------------

    denominator = (
        gt_id_count
        + tracker_id_count
        - potential_matches
    )

    global_alignment = np.divide(
        potential_matches,
        denominator,
        out=np.zeros_like(
            potential_matches
        ),
        where=(
            denominator
            > EPS
        ),
    )

    match_counts = [
        np.zeros_like(
            potential_matches
        )
        for _ in ALPHAS
    ]

    # -----------------------------------------------------
    # Second pass
    # -----------------------------------------------------

    for (
        gt_ids_frame,
        tracker_ids_frame,
        similarity,
    ) in zip(
        data["gt_ids"],
        data["tr_ids"],
        data["sims"],
    ):

        # No GT this frame

        if len(
            gt_ids_frame
        ) == 0:

            false_positive += len(
                tracker_ids_frame
            )

            continue

        # No predictions this frame

        if len(
            tracker_ids_frame
        ) == 0:

            false_negative += len(
                gt_ids_frame
            )

            continue

        # HOTA matching score:
        #
        # global association quality
        # *
        # current IoU

        score_matrix = (
            global_alignment[
                gt_ids_frame[:, None],
                tracker_ids_frame[
                    None,
                    :
                ],
            ]
            * similarity
        )

        (
            match_rows,
            match_cols,
        ) = linear_sum_assignment(
            -score_matrix
        )

        # -------------------------------------------------
        # Evaluate all HOTA alpha thresholds
        # -------------------------------------------------

        for (
            alpha_index,
            alpha,
        ) in enumerate(
            ALPHAS
        ):

            matched_mask = (
                similarity[
                    match_rows,
                    match_cols,
                ]
                >= alpha - EPS
            )

            rows = match_rows[
                matched_mask
            ]

            cols = match_cols[
                matched_mask
            ]

            num_matches = len(
                rows
            )

            true_positive[
                alpha_index
            ] += num_matches

            false_negative[
                alpha_index
            ] += (
                len(
                    gt_ids_frame
                )
                - num_matches
            )

            false_positive[
                alpha_index
            ] += (
                len(
                    tracker_ids_frame
                )
                - num_matches
            )

            if num_matches:

                localization_accuracy[
                    alpha_index
                ] += similarity[
                    rows,
                    cols,
                ].sum()

                match_counts[
                    alpha_index
                ][
                    gt_ids_frame[
                        rows
                    ],
                    tracker_ids_frame[
                        cols
                    ],
                ] += 1

    # -----------------------------------------------------
    # Association Accuracy
    # -----------------------------------------------------

    for alpha_index in range(
        num_thresholds
    ):

        matches = match_counts[
            alpha_index
        ]

        pair_accuracy = (
            matches
            / np.maximum(
                1.0,
                (
                    gt_id_count
                    + tracker_id_count
                    - matches
                ),
            )
        )

        association_accuracy[
            alpha_index
        ] = (
            np.sum(
                matches
                * pair_accuracy
            )
            / max(
                1.0,
                true_positive[
                    alpha_index
                ],
            )
        )

    localization_accuracy = (
        np.maximum(
            1e-10,
            localization_accuracy,
        )
        / np.maximum(
            1e-10,
            true_positive,
        )
    )

    return finish_hota(
        true_positive,
        false_negative,
        false_positive,
        association_accuracy,
        localization_accuracy,
    )


# =========================================================
# 8. CLEAR MOT
#
# MOTA
# IDSW
# Fragmentation
# Precision
# Recall
# MOTP
# =========================================================

def eval_clear(
    data,
    threshold=0.5,
):

    true_positive = 0
    false_negative = 0
    false_positive = 0

    id_switches = 0
    fragmentation = 0

    motp_sum = 0.0

    # -----------------------------------------------------
    # No Ground Truth
    # -----------------------------------------------------

    if data["n_gt_ids"] == 0:

        return {
            "TP": 0,

            "FN": 0,

            "FP": (
                data["n_tr_dets"]
            ),

            "IDSW": 0,

            "Frag": 0,

            "MOTA": 0.0,

            "MOTP": 0.0,

            "Recall": 0.0,

            "Precision": 0.0,
        }

    # -----------------------------------------------------
    # No predictions
    # -----------------------------------------------------

    if data["n_tr_dets"] == 0:

        return {
            "TP": 0,

            "FN": (
                data["n_gt_dets"]
            ),

            "FP": 0,

            "IDSW": 0,

            "Frag": 0,

            "MOTA": 0.0,

            "MOTP": 0.0,

            "Recall": 0.0,

            "Precision": 0.0,
        }

    num_gt_ids = data[
        "n_gt_ids"
    ]

    # 每个 GT ID 出现多少次

    gt_id_count = np.zeros(
        num_gt_ids
    )

    # 每个 GT ID 成功匹配多少次

    matched_count = np.zeros(
        num_gt_ids
    )

    # fragmentation 统计

    fragmentation_count = np.zeros(
        num_gt_ids
    )

    # GT ID 上一次被分配的 tracker ID。
    #
    # 即使中间断了几帧，也保留，
    # 用于统计 IDSW。

    previous_tracker_id = np.full(
        num_gt_ids,
        np.nan,
    )

    # 只保存上一帧 tracker ID。
    #
    # 用于优先保持连续匹配。

    previous_frame_tracker_id = np.full(
        num_gt_ids,
        np.nan,
    )

    # -----------------------------------------------------
    # Frame-by-frame matching
    # -----------------------------------------------------

    for (
        gt_ids_frame,
        tracker_ids_frame,
        similarity,
    ) in zip(
        data["gt_ids"],
        data["tr_ids"],
        data["sims"],
    ):

        # -------------------------------------------------
        # No GT
        # -------------------------------------------------

        if len(
            gt_ids_frame
        ) == 0:

            false_positive += len(
                tracker_ids_frame
            )

            previous_frame_tracker_id[:] = (
                np.nan
            )

            continue

        # -------------------------------------------------
        # No predictions
        # -------------------------------------------------

        if len(
            tracker_ids_frame
        ) == 0:

            false_negative += len(
                gt_ids_frame
            )

            gt_id_count[
                gt_ids_frame
            ] += 1

            previous_frame_tracker_id[:] = (
                np.nan
            )

            continue

        # -------------------------------------------------
        # Prefer keeping same ID from previous frame
        # -------------------------------------------------

        continuation = (
            tracker_ids_frame[
                None,
                :
            ]
            == previous_frame_tracker_id[
                gt_ids_frame[
                    :,
                    None
                ]
            ]
        ).astype(
            float
        )

        # 1000 确保：
        #
        # 先保证上一帧 ID 连续，
        # 再比较 IoU。

        score_matrix = (
            1000.0
            * continuation
            + similarity
        )

        # IoU < threshold
        # 不允许匹配。

        score_matrix[
            similarity
            < threshold - EPS
        ] = 0.0

        (
            match_rows,
            match_cols,
        ) = linear_sum_assignment(
            -score_matrix
        )

        matched_mask = (
            score_matrix[
                match_rows,
                match_cols,
            ]
            > EPS
        )

        match_rows = match_rows[
            matched_mask
        ]

        match_cols = match_cols[
            matched_mask
        ]

        matched_gt_ids = (
            gt_ids_frame[
                match_rows
            ]
        )

        matched_tracker_ids = (
            tracker_ids_frame[
                match_cols
            ]
        )

        # -------------------------------------------------
        # ID Switch
        # -------------------------------------------------

        if len(
            matched_gt_ids
        ):

            old_tracker_ids = (
                previous_tracker_id[
                    matched_gt_ids
                ]
            )

            is_switch = (
                ~np.isnan(
                    old_tracker_ids
                )
            ) & (
                old_tracker_ids
                != matched_tracker_ids
            )

            id_switches += int(
                np.sum(
                    is_switch
                )
            )

        # -------------------------------------------------
        # Update tracking statistics
        # -------------------------------------------------

        gt_id_count[
            gt_ids_frame
        ] += 1

        matched_count[
            matched_gt_ids
        ] += 1

        was_untracked = np.isnan(
            previous_frame_tracker_id
        )

        previous_tracker_id[
            matched_gt_ids
        ] = (
            matched_tracker_ids
        )

        previous_frame_tracker_id[
            :
        ] = np.nan

        previous_frame_tracker_id[
            matched_gt_ids
        ] = (
            matched_tracker_ids
        )

        currently_tracked = (
            ~np.isnan(
                previous_frame_tracker_id
            )
        )

        fragmentation_count += (
            was_untracked
            & currently_tracked
        )

        # -------------------------------------------------
        # TP / FN / FP
        # -------------------------------------------------

        num_matches = len(
            matched_gt_ids
        )

        true_positive += (
            num_matches
        )

        false_negative += (
            len(
                gt_ids_frame
            )
            - num_matches
        )

        false_positive += (
            len(
                tracker_ids_frame
            )
            - num_matches
        )

        if num_matches:

            motp_sum += float(
                similarity[
                    match_rows,
                    match_cols,
                ].sum()
            )

    # 每一条轨迹第一次出现不算 fragmentation，
    # 所以减1。

    fragmentation = int(
        np.sum(
            fragmentation_count[
                fragmentation_count
                > 0
            ]
            - 1
        )
    )

    gt_detection_count = max(
        1.0,
        (
            true_positive
            + false_negative
        ),
    )

    # -----------------------------------------------------
    # MOTA
    #
    # 1 - (FN + FP + IDSW) / GT
    #
    # 等价于：
    #
    # (TP - FP - IDSW) / GT
    # -----------------------------------------------------

    mota = (
        (
            true_positive
            - false_positive
            - id_switches
        )
        / gt_detection_count
    )

    motp = (
        motp_sum
        / max(
            1.0,
            true_positive,
        )
    )

    recall = (
        true_positive
        / gt_detection_count
    )

    precision = (
        true_positive
        / max(
            1.0,
            (
                true_positive
                + false_positive
            ),
        )
    )

    return {
        "TP": int(
            true_positive
        ),

        "FN": int(
            false_negative
        ),

        "FP": int(
            false_positive
        ),

        "IDSW": int(
            id_switches
        ),

        "Frag": (
            fragmentation
        ),

        "MOTA": mota,

        "MOTP": motp,

        "Recall": recall,

        "Precision": precision,
    }


# =========================================================
# 9. Identity Metrics
#
# IDTP
# IDFP
# IDFN
# IDP
# IDR
# IDF1
# =========================================================

def eval_identity(
    data,
    threshold=0.5,
):

    # -----------------------------------------------------
    # Empty prediction
    # -----------------------------------------------------

    if data["n_tr_dets"] == 0:

        return {
            "IDTP": 0,

            "IDFN": (
                data["n_gt_dets"]
            ),

            "IDFP": 0,

            "IDF1": 0.0,

            "IDP": 0.0,

            "IDR": 0.0,
        }

    # -----------------------------------------------------
    # Empty GT
    # -----------------------------------------------------

    if data["n_gt_dets"] == 0:

        return {
            "IDTP": 0,

            "IDFN": 0,

            "IDFP": (
                data["n_tr_dets"]
            ),

            "IDF1": 0.0,

            "IDP": 0.0,

            "IDR": 0.0,
        }

    num_gt_ids = (
        data["n_gt_ids"]
    )

    num_tracker_ids = (
        data["n_tr_ids"]
    )

    potential_matches = np.zeros(
        (
            num_gt_ids,
            num_tracker_ids,
        )
    )

    gt_count = np.zeros(
        num_gt_ids
    )

    tracker_count = np.zeros(
        num_tracker_ids
    )

    # -----------------------------------------------------
    # Count all possible GT / tracker trajectory matches
    # -----------------------------------------------------

    for (
        gt_ids_frame,
        tracker_ids_frame,
        similarity,
    ) in zip(
        data["gt_ids"],
        data["tr_ids"],
        data["sims"],
    ):

        if (
            len(gt_ids_frame)
            and len(
                tracker_ids_frame
            )
        ):

            (
                rows,
                cols,
            ) = np.nonzero(
                similarity
                >= threshold - EPS
            )

            if len(
                rows
            ):

                np.add.at(
                    potential_matches,
                    (
                        gt_ids_frame[
                            rows
                        ],
                        tracker_ids_frame[
                            cols
                        ],
                    ),
                    1,
                )

        if len(
            gt_ids_frame
        ):

            np.add.at(
                gt_count,
                gt_ids_frame,
                1,
            )

        if len(
            tracker_ids_frame
        ):

            np.add.at(
                tracker_count,
                tracker_ids_frame,
                1,
            )

    # -----------------------------------------------------
    # Global trajectory assignment
    # -----------------------------------------------------

    matrix_size = (
        num_gt_ids
        + num_tracker_ids
    )

    false_positive_matrix = np.zeros(
        (
            matrix_size,
            matrix_size,
        )
    )

    false_negative_matrix = np.zeros(
        (
            matrix_size,
            matrix_size,
        )
    )

    # Dummy assignments

    false_positive_matrix[
        num_gt_ids:,
        :num_tracker_ids,
    ] = 1e10

    false_negative_matrix[
        :num_gt_ids,
        num_tracker_ids:,
    ] = 1e10

    # GT costs

    for gt_id in range(
        num_gt_ids
    ):

        false_negative_matrix[
            gt_id,
            :num_tracker_ids,
        ] = gt_count[
            gt_id
        ]

        false_negative_matrix[
            gt_id,
            num_tracker_ids
            + gt_id,
        ] = gt_count[
            gt_id
        ]

    # Tracker costs

    for tracker_id in range(
        num_tracker_ids
    ):

        false_positive_matrix[
            :num_gt_ids,
            tracker_id,
        ] = tracker_count[
            tracker_id
        ]

        false_positive_matrix[
            num_gt_ids
            + tracker_id,
            tracker_id,
        ] = tracker_count[
            tracker_id
        ]

    false_negative_matrix[
        :num_gt_ids,
        :num_tracker_ids,
    ] -= potential_matches

    false_positive_matrix[
        :num_gt_ids,
        :num_tracker_ids,
    ] -= potential_matches

    (
        rows,
        cols,
    ) = linear_sum_assignment(
        false_negative_matrix
        + false_positive_matrix
    )

    identity_false_negative = int(
        round(
            false_negative_matrix[
                rows,
                cols,
            ].sum()
        )
    )

    identity_false_positive = int(
        round(
            false_positive_matrix[
                rows,
                cols,
            ].sum()
        )
    )

    identity_true_positive = int(
        round(
            gt_count.sum()
            - identity_false_negative
        )
    )

    # -----------------------------------------------------
    # ID Precision
    # -----------------------------------------------------

    identity_precision = (
        identity_true_positive
        / max(
            1.0,
            (
                identity_true_positive
                + identity_false_positive
            ),
        )
    )

    # -----------------------------------------------------
    # ID Recall
    # -----------------------------------------------------

    identity_recall = (
        identity_true_positive
        / max(
            1.0,
            (
                identity_true_positive
                + identity_false_negative
            ),
        )
    )

    # -----------------------------------------------------
    # IDF1
    # -----------------------------------------------------

    identity_f1 = (
        2
        * identity_true_positive
        / max(
            1.0,
            (
                2
                * identity_true_positive
                + identity_false_negative
                + identity_false_positive
            ),
        )
    )

    return {
        "IDTP": (
            identity_true_positive
        ),

        "IDFN": (
            identity_false_negative
        ),

        "IDFP": (
            identity_false_positive
        ),

        "IDF1": (
            identity_f1
        ),

        "IDP": (
            identity_precision
        ),

        "IDR": (
            identity_recall
        ),
    }


# =========================================================
# 10. Full evaluation
# =========================================================

def evaluate(
    gt_path: Path,
    pred_path: Path,
    num_frames: int,
    classes: Sequence[int],
    match_iou: float,
):

    ground_truth = read_rows(
        gt_path
    )

    predictions = read_rows(
        pred_path
    )

    # 如果没有显式给 --frames，
    # 从 GT / prediction 最大帧号推断。

    if num_frames <= 0:

        num_frames = max(
            [0]
            + [
                int(row[0])
                for row
                in ground_truth
            ]
            + [
                int(row[0])
                for row
                in predictions
            ]
        )

    if num_frames <= 0:
        raise ValueError(
            "Cannot infer number of frames "
            "from empty GT and prediction files"
        )

    data = build_data(
        ground_truth,
        predictions,
        num_frames,
        classes,
    )

    hota_results = eval_hota(
        data
    )

    clear_results = eval_clear(
        data,
        threshold=match_iou,
    )

    identity_results = eval_identity(
        data,
        threshold=match_iou,
    )

    # TrackEval HOTA summary:
    #
    # HOTA = mean over alpha thresholds.

    results = {
        "HOTA": (
            100
            * float(
                np.mean(
                    hota_results[
                        "HOTA"
                    ]
                )
            )
        ),

        "DetA": (
            100
            * float(
                np.mean(
                    hota_results[
                        "DetA"
                    ]
                )
            )
        ),

        "AssA": (
            100
            * float(
                np.mean(
                    hota_results[
                        "AssA"
                    ]
                )
            )
        ),

        "LocA": (
            100
            * float(
                np.mean(
                    hota_results[
                        "LocA"
                    ]
                )
            )
        ),

        "IDF1": (
            100
            * identity_results[
                "IDF1"
            ]
        ),

        "IDP": (
            100
            * identity_results[
                "IDP"
            ]
        ),

        "IDR": (
            100
            * identity_results[
                "IDR"
            ]
        ),

        "MOTA": (
            100
            * clear_results[
                "MOTA"
            ]
        ),

        "MOTP": (
            100
            * clear_results[
                "MOTP"
            ]
        ),

        "Recall": (
            100
            * clear_results[
                "Recall"
            ]
        ),

        "Precision": (
            100
            * clear_results[
                "Precision"
            ]
        ),

        "IDSW": (
            clear_results[
                "IDSW"
            ]
        ),

        "Frag": (
            clear_results[
                "Frag"
            ]
        ),

        "GT_Dets": (
            data[
                "n_gt_dets"
            ]
        ),

        "Tracker_Dets": (
            data[
                "n_tr_dets"
            ]
        ),

        "Frames": (
            num_frames
        ),
    }

    return results


# =========================================================
# 11. Command-line arguments
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VisDrone MOT "
            "tracking results"
        )
    )

    parser.add_argument(
        "--gt",
        type=Path,
        required=True,
        help=(
            "VisDrone ground-truth txt"
        ),
    )

    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help=(
            "Tracker result txt"
        ),
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help=(
            "Sequence frame count; "
            "0 = infer from GT/pred"
        ),
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=list(
            VALID_CLASSES
        ),
        help=(
            "VisDrone class IDs. "
            "Default = 1 2 3 4 5 "
            "6 7 8 9 10"
        ),
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help=(
            "IoU threshold for "
            "MOTA / IDF1 / IDSW"
        ),
    )

    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help=(
            "Optional JSON output path"
        ),
    )

    return parser.parse_args()


# =========================================================
# 12. Main
# =========================================================

def main():

    args = parse_args()

    if not args.gt.is_file():
        raise FileNotFoundError(
            args.gt
        )

    if not args.pred.is_file():
        raise FileNotFoundError(
            args.pred
        )

    if not (
        0
        < args.match_iou
        <= 1
    ):
        raise ValueError(
            "--match-iou must "
            "be in (0, 1]"
        )

    invalid_classes = sorted(
        set(
            args.classes
        )
        - set(
            VALID_CLASSES
        )
    )

    if invalid_classes:
        raise ValueError(
            "Invalid VisDrone "
            f"class IDs: "
            f"{invalid_classes}; "
            "valid IDs are 1..10"
        )

    metrics = evaluate(
        gt_path=args.gt,
        pred_path=args.pred,
        num_frames=args.frames,
        classes=args.classes,
        match_iou=args.match_iou,
    )

    print()
    print("=" * 52)
    print(
        "VisDrone MOT Evaluation"
    )
    print("=" * 52)

    print(
        f"HOTA      : "
        f"{metrics['HOTA']:.2f}"
    )

    print(
        f"DetA      : "
        f"{metrics['DetA']:.2f}"
    )

    print(
        f"AssA      : "
        f"{metrics['AssA']:.2f}"
    )

    print(
        f"IDF1      : "
        f"{metrics['IDF1']:.2f}"
    )

    print(
        f"MOTA      : "
        f"{metrics['MOTA']:.2f}"
    )

    print(
        f"IDSW      : "
        f"{metrics['IDSW']}"
    )

    print(
        f"Frag      : "
        f"{metrics['Frag']}"
    )

    print(
        f"Recall    : "
        f"{metrics['Recall']:.2f}"
    )

    print(
        f"Precision : "
        f"{metrics['Precision']:.2f}"
    )

    print(
        f"GT dets   : "
        f"{metrics['GT_Dets']}"
    )

    print(
        f"Pred dets : "
        f"{metrics['Tracker_Dets']}"
    )

    print(
        f"Frames    : "
        f"{metrics['Frames']}"
    )

    if args.save:

        args.save.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.save.write_text(
            json.dumps(
                metrics,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            f"Saved     : "
            f"{args.save}"
        )


if __name__ == "__main__":
    main()