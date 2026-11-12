from pathlib import Path
from ultralytics import YOLO


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "weights" / "yolo11n.pt"
DATA_CONFIG = PROJECT_ROOT / "configs" / "visdrone.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "train"


def main():

    print("=" * 60)
    print("YOLO11n VisDrone Baseline Training")
    print("=" * 60)

    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {DATA_CONFIG}")
    print(f"Output: {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not DATA_CONFIG.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_CONFIG}"
        )

    # --------------------------------------------------------
    # Load pretrained YOLO11n
    # --------------------------------------------------------

    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # Train
    #
    # First run:
    # epochs=1
    #
    # This is only used to verify the dataset pipeline.
    # Do NOT start long training before this works.
    # --------------------------------------------------------

    results = model.train(
        data=str(DATA_CONFIG),

        # Basic training settings
        epochs=100,
        imgsz=640,
        batch=16,

        # GPU
        device=0,
        workers=8,

        # Training
        pretrained=True,
        patience=20,

        # Dense VisDrone scenes
        max_det=1000,

        # Output
        project=str(OUTPUT_DIR),
        name="yolo11n_visdrone_baseline",
        exist_ok=False,

        save=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Training finished.")
    print("=" * 60)

    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR / 'yolo11n_visdrone_test'}"
    )

    return results


if __name__ == "__main__":
    main()