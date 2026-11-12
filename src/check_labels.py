from pathlib import Path
import random

import cv2


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets" / "VisDrone"

IMAGE_DIR = DATASET_ROOT / "train" / "images"
LABEL_DIR = DATASET_ROOT / "train" / "labels"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "label_check"


# ============================================================
# VisDrone class names
# YOLO class_id: 0 ~ 9
# ============================================================

CLASS_NAMES = [
    "pedestrian",        # 0
    "people",            # 1
    "bicycle",           # 2
    "car",               # 3
    "van",               # 4
    "truck",             # 5
    "tricycle",          # 6
    "awning-tricycle",   # 7
    "bus",               # 8
    "motor",             # 9
]


def draw_yolo_labels(image_path, label_path, output_path):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    image_height, image_width = image.shape[:2]

    if not label_path.exists():
        print(f"Missing label: {label_path}")
        return

    with open(label_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                print(
                    f"Invalid label format: "
                    f"{label_path} -> {line}"
                )
                continue

            class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            box_width = float(values[3])
            box_height = float(values[4])

            # ------------------------------------------------
            # Convert normalized YOLO coordinates
            # back to pixel coordinates
            # ------------------------------------------------

            x_center *= image_width
            y_center *= image_height

            box_width *= image_width
            box_height *= image_height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)

            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Prevent coordinates from exceeding image bounds
            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))

            x2 = max(0, min(x2, image_width - 1))
            y2 = max(0, min(y2, image_height - 1))

            # ------------------------------------------------
            # Class name
            # ------------------------------------------------

            if 0 <= class_id < len(CLASS_NAMES):
                class_name = CLASS_NAMES[class_id]
            else:
                class_name = f"class_{class_id}"

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Draw class name
            # ------------------------------------------------

            cv2.putText(
                image,
                class_name,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

    cv2.imwrite(str(output_path), image)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = list(IMAGE_DIR.glob("*.jpg"))

    if not image_files:
        raise RuntimeError(
            f"No images found in: {IMAGE_DIR}"
        )

    # Randomly check 10 images
    sample_number = min(
        10,
        len(image_files)
    )

    random.seed(42)

    selected_images = random.sample(
        image_files,
        sample_number
    )

    print("=" * 60)
    print("VisDrone YOLO Label Visualization")
    print("=" * 60)

    print(f"Image directory : {IMAGE_DIR}")
    print(f"Label directory : {LABEL_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    print(f"\nChecking {sample_number} images...\n")

    for index, image_path in enumerate(
        selected_images,
        start=1
    ):

        label_path = (
            LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        output_path = (
            OUTPUT_DIR
            / image_path.name
        )

        draw_yolo_labels(
            image_path,
            label_path,
            output_path
        )

        print(
            f"[{index}/{sample_number}] "
            f"{image_path.name}"
        )

    print("\nFinished.")
    print(
        f"Check visualization results in:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()