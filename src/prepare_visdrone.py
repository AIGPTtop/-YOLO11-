from pathlib import Path
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "VisDrone"


def convert_split(split):
    """
    Convert VisDrone annotations to YOLO format.
    """

    split_dir = DATASET_ROOT / split

    image_dir = split_dir / "images"
    annotation_dir = split_dir / "annotations"
    label_dir = split_dir / "labels"

    label_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.glob("*.jpg"))

    print(f"\nProcessing {split}")
    print(f"Images: {len(image_files)}")

    converted_images = 0
    converted_objects = 0
    ignored_objects = 0

    for image_path in image_files:

        annotation_path = (
            annotation_dir / f"{image_path.stem}.txt"
        )

        label_path = (
            label_dir / f"{image_path.stem}.txt"
        )

        if not annotation_path.exists():
            print(f"Missing annotation: {annotation_path}")
            continue

        # Read image size
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        yolo_labels = []

        with open(
            annotation_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                values = line.split(",")

                if len(values) < 8:
                    continue

                x = float(values[0])
                y = float(values[1])
                w = float(values[2])
                h = float(values[3])

                category = int(values[5])

                # VisDrone category 0 = ignored regions
                # VisDrone categories:
                #
                # 0  = ignored regions
                # 1  = pedestrian
                # 2  = people
                # 3  = bicycle
                # 4  = car
                # 5  = van
                # 6  = truck
                # 7  = tricycle
                # 8  = awning-tricycle
                # 9  = bus
                # 10 = motor
                # 11 = others
                #
                # Only categories 1~10 are used in our YOLO training.

                if category < 1 or category > 10:
                    ignored_objects += 1
                    continue

                class_id = category - 1

                # Convert top-left xywh
                # to normalized center xywh
                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height

                width = w / image_width
                height = h / image_height

                # Basic validity check
                if (
                    width <= 0
                    or height <= 0
                    or x_center < 0
                    or x_center > 1
                    or y_center < 0
                    or y_center > 1
                ):
                    continue

                yolo_labels.append(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

                converted_objects += 1

        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("\n".join(yolo_labels))

        converted_images += 1

    print(f"Converted images: {converted_images}")
    print(f"Converted objects: {converted_objects}")
    print(f"Ignored objects: {ignored_objects}")


def main():

    for split in ["train", "val", "test"]:
        convert_split(split)

    print("\nVisDrone conversion finished.")


if __name__ == "__main__":
    main()