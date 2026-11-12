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

        epochs=100,
        imgsz=960,
        batch=32,

        device=0,
        workers=8,

        pretrained=True,
        patience=20,
        max_det=1000,

        project=str(OUTPUT_DIR),
        name="yolo11n_visdrone_imgsz960",

        save=True,
        plots=True,
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