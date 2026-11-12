from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets" / "VisDrone"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# VisDrone classes
# ============================================================

CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}


# ============================================================
# COCO-style object size thresholds
#
# Small:  area < 32^2
# Medium: 32^2 <= area < 96^2
# Large:  area >= 96^2
#
# IMPORTANT:
# Area is measured in ORIGINAL IMAGE pixels.
# ============================================================

SMALL_THRESHOLD = 32 ** 2
MEDIUM_THRESHOLD = 96 ** 2


def classify_object_size(width, height):
    """Classify object by pixel area using COCO-style thresholds."""

    area = width * height

    if area < SMALL_THRESHOLD:
        return "small"

    elif area < MEDIUM_THRESHOLD:
        return "medium"

    else:
        return "large"


def analyze_split(split_name):
    """
    Analyze one dataset split.

    Expected directory:
        VisDrone/
            train/
                images/
                labels/

            val/
                images/
                labels/
    """

    image_dir = DATASET_ROOT / split_name / "images"
    label_dir = DATASET_ROOT / split_name / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Label directory not found: {label_dir}"
        )

    # --------------------------------------------------------
    # Statistics containers
    # --------------------------------------------------------

    class_counter = Counter()

    size_counter = Counter({
        "small": 0,
        "medium": 0,
        "large": 0,
    })

    objects_per_image = []

    object_widths = []
    object_heights = []
    object_areas = []

    image_widths = []
    image_heights = []

    invalid_labels = 0
    empty_images = 0

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    image_files = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    print("\n" + "=" * 70)
    print(f"Analyzing split: {split_name}")
    print("=" * 70)

    print(f"Images found: {len(image_files)}")

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    for index, image_path in enumerate(image_files, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARNING] Cannot read image: {image_path}")
            continue

        img_h, img_w = image.shape[:2]

        image_widths.append(img_w)
        image_heights.append(img_h)

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            objects_per_image.append(0)
            empty_images += 1
            continue

        lines = label_path.read_text(
            encoding="utf-8"
        ).strip().splitlines()

        object_count = 0

        for line in lines:

            if not line.strip():
                continue

            parts = line.strip().split()

            if len(parts) != 5:
                invalid_labels += 1
                continue

            try:
                class_id = int(float(parts[0]))

                x_center = float(parts[1])
                y_center = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])

            except ValueError:
                invalid_labels += 1
                continue

            # Check class range
            if class_id not in CLASS_NAMES:
                invalid_labels += 1
                continue

            # Check normalized coordinates
            if (
                width_norm <= 0
                or height_norm <= 0
                or width_norm > 1
                or height_norm > 1
            ):
                invalid_labels += 1
                continue

            # ------------------------------------------------
            # Convert normalized YOLO width/height
            # back to original-image pixels
            # ------------------------------------------------

            bbox_width = width_norm * img_w
            bbox_height = height_norm * img_h

            bbox_area = bbox_width * bbox_height

            # ------------------------------------------------
            # Class statistics
            # ------------------------------------------------

            class_counter[class_id] += 1

            # ------------------------------------------------
            # Object size statistics
            # ------------------------------------------------

            size_type = classify_object_size(
                bbox_width,
                bbox_height,
            )

            size_counter[size_type] += 1

            object_widths.append(bbox_width)
            object_heights.append(bbox_height)
            object_areas.append(bbox_area)

            object_count += 1

        objects_per_image.append(object_count)

        if object_count == 0:
            empty_images += 1

        if index % 500 == 0:
            print(
                f"Processed {index}/{len(image_files)} images..."
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_objects = sum(class_counter.values())

    print("\n" + "-" * 70)
    print(f"{split_name.upper()} SUMMARY")
    print("-" * 70)

    print(f"Images           : {len(image_files)}")
    print(f"Objects          : {total_objects}")
    print(f"Empty images     : {empty_images}")
    print(f"Invalid labels   : {invalid_labels}")

    if objects_per_image:
        print(
            f"Objects / image  : "
            f"{np.mean(objects_per_image):.2f}"
        )

        print(
            f"Median objects   : "
            f"{np.median(objects_per_image):.2f}"
        )

        print(
            f"Max objects/image: "
            f"{np.max(objects_per_image)}"
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass Distribution")
    print("-" * 70)

    for class_id, class_name in CLASS_NAMES.items():

        count = class_counter[class_id]

        percentage = (
            count / total_objects * 100
            if total_objects > 0
            else 0
        )

        print(
            f"{class_id:2d} "
            f"{class_name:<18} "
            f"{count:8d} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Object scale distribution
    # --------------------------------------------------------

    print("\nObject Scale Distribution")
    print("-" * 70)

    for size_name in ["small", "medium", "large"]:

        count = size_counter[size_name]

        percentage = (
            count / total_objects * 100
            if total_objects > 0
            else 0
        )

        print(
            f"{size_name:<10} "
            f"{count:8d} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Bounding box statistics
    # --------------------------------------------------------

    if object_widths:

        print("\nBounding Box Statistics (pixels)")
        print("-" * 70)

        print(
            f"Mean width       : "
            f"{np.mean(object_widths):.2f}"
        )

        print(
            f"Mean height      : "
            f"{np.mean(object_heights):.2f}"
        )

        print(
            f"Median width     : "
            f"{np.median(object_widths):.2f}"
        )

        print(
            f"Median height    : "
            f"{np.median(object_heights):.2f}"
        )

        print(
            f"Median area      : "
            f"{np.median(object_areas):.2f}"
        )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "split": split_name,
        "num_images": len(image_files),
        "total_objects": total_objects,

        "class_counter": class_counter,
        "size_counter": size_counter,

        "objects_per_image": objects_per_image,

        "object_widths": object_widths,
        "object_heights": object_heights,
        "object_areas": object_areas,

        "invalid_labels": invalid_labels,
        "empty_images": empty_images,
    }


def plot_class_distribution(result):
    """Save class distribution plot."""

    class_names = [
        CLASS_NAMES[i]
        for i in range(len(CLASS_NAMES))
    ]

    counts = [
        result["class_counter"][i]
        for i in range(len(CLASS_NAMES))
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(
        class_names,
        counts,
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.ylabel("Number of Objects")
    plt.xlabel("Class")

    plt.title(
        f"VisDrone {result['split']} - Class Distribution"
    )

    plt.tight_layout()

    save_path = (
        OUTPUT_DIR
        / f"{result['split']}_class_distribution.png"
    )

    plt.savefig(
        save_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {save_path}")


def plot_size_distribution(result):
    """Save small/medium/large distribution plot."""

    size_names = [
        "small",
        "medium",
        "large",
    ]

    counts = [
        result["size_counter"][name]
        for name in size_names
    ]

    plt.figure(figsize=(8, 6))

    plt.bar(
        size_names,
        counts,
    )

    plt.ylabel("Number of Objects")
    plt.xlabel("Object Size")

    plt.title(
        f"VisDrone {result['split']} - Object Scale Distribution"
    )

    plt.tight_layout()

    save_path = (
        OUTPUT_DIR
        / f"{result['split']}_size_distribution.png"
    )

    plt.savefig(
        save_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {save_path}")


def plot_objects_per_image(result):
    """Save object-density histogram."""

    values = result["objects_per_image"]

    if not values:
        return

    plt.figure(figsize=(9, 6))

    plt.hist(
        values,
        bins=50,
    )

    plt.xlabel("Objects per Image")
    plt.ylabel("Number of Images")

    plt.title(
        f"VisDrone {result['split']} - Object Density"
    )

    plt.tight_layout()

    save_path = (
        OUTPUT_DIR
        / f"{result['split']}_objects_per_image.png"
    )

    plt.savefig(
        save_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {save_path}")


def plot_bbox_area_distribution(result):
    """Save bounding-box area histogram."""

    areas = np.array(
        result["object_areas"],
        dtype=np.float64,
    )

    if len(areas) == 0:
        return

    # log10 is useful because VisDrone object areas
    # have a very wide range.
    log_areas = np.log10(
        np.maximum(areas, 1.0)
    )

    plt.figure(figsize=(9, 6))

    plt.hist(
        log_areas,
        bins=60,
    )

    plt.xlabel("log10(Bounding Box Area / pixel^2)")
    plt.ylabel("Number of Objects")

    plt.title(
        f"VisDrone {result['split']} - Bounding Box Area"
    )

    plt.tight_layout()

    save_path = (
        OUTPUT_DIR
        / f"{result['split']}_bbox_area.png"
    )

    plt.savefig(
        save_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {save_path}")


def save_summary(train_result, val_result):
    """Save key statistics to a text file."""

    summary_path = OUTPUT_DIR / "dataset_summary.txt"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for result in [train_result, val_result]:

            total = result["total_objects"]

            f.write("=" * 70 + "\n")
            f.write(
                f"{result['split'].upper()} DATASET\n"
            )
            f.write("=" * 70 + "\n")

            f.write(
                f"Images: {result['num_images']}\n"
            )

            f.write(
                f"Objects: {total}\n"
            )

            f.write(
                f"Invalid labels: "
                f"{result['invalid_labels']}\n"
            )

            f.write(
                f"Empty images: "
                f"{result['empty_images']}\n"
            )

            if result["objects_per_image"]:

                f.write(
                    "Average objects/image: "
                    f"{np.mean(result['objects_per_image']):.2f}\n"
                )

                f.write(
                    "Maximum objects/image: "
                    f"{np.max(result['objects_per_image'])}\n"
                )

            f.write("\nClass Distribution:\n")

            for class_id, class_name in CLASS_NAMES.items():

                count = result["class_counter"][class_id]

                percentage = (
                    count / total * 100
                    if total > 0
                    else 0
                )

                f.write(
                    f"{class_name:<18} "
                    f"{count:8d} "
                    f"{percentage:6.2f}%\n"
                )

            f.write("\nObject Scale Distribution:\n")

            for size_name in [
                "small",
                "medium",
                "large",
            ]:

                count = result["size_counter"][size_name]

                percentage = (
                    count / total * 100
                    if total > 0
                    else 0
                )

                f.write(
                    f"{size_name:<10} "
                    f"{count:8d} "
                    f"{percentage:6.2f}%\n"
                )

            f.write("\n\n")

    print(f"\nSummary saved to: {summary_path}")


def main():

    print("=" * 70)
    print("VisDrone Dataset Analysis")
    print("=" * 70)

    print(f"Dataset: {DATASET_ROOT}")
    print(f"Output : {OUTPUT_DIR}")

    # ========================================================
    # Analyze train / val
    # ========================================================

    train_result = analyze_split("train")
    val_result = analyze_split("val")

    # ========================================================
    # Visualization
    # ========================================================

    for result in [
        train_result,
        val_result,
    ]:

        plot_class_distribution(result)
        plot_size_distribution(result)
        plot_objects_per_image(result)
        plot_bbox_area_distribution(result)

    # ========================================================
    # Save summary
    # ========================================================

    save_summary(
        train_result,
        val_result,
    )

    print("\n" + "=" * 70)
    print("Dataset analysis finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()