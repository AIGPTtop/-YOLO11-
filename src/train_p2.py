from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_CONFIG = PROJECT_ROOT / "configs" / "yolo11n_p2.yaml"
PRETRAINED = PROJECT_ROOT / "weights" / "yolo11n.pt"
DATA_CONFIG = PROJECT_ROOT / "configs" / "visdrone.yaml"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "train"


def main():

    print("=" * 70)
    print("Experiment E3")
    print("YOLO11n-P2 + VisDrone + imgsz=960")
    print("=" * 70)

    # Build P2 architecture
    model = YOLO(str(MODEL_CONFIG))

    # Transfer compatible YOLO11n pretrained weights
    model.load(str(PRETRAINED))

    # Formal E3
    model.train(
        data=str(DATA_CONFIG),

        epochs=100,
        imgsz=960,

        # IMPORTANT:
        # Set this to exactly the same value used in E1.
        batch=32,

        device=0,
        workers=8,

        patience=20,

        # Keep the training strategy aligned with E1.
        # No E2 custom augmentation here.

        max_det=1000,

        project=str(OUTPUT_DIR),
        name="yolo11n_p2_visdrone_imgsz960",

        exist_ok=False,

        save=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()