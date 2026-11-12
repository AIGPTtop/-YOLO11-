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

    print("=" * 70)
    print("Experiment E2")
    print("YOLO11n + 960 + VisDrone-oriented Data Augmentation")
    print("=" * 70)

    print(f"Model : {MODEL_PATH}")
    print(f"Data  : {DATA_CONFIG}")
    print(f"Output: {OUTPUT_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not DATA_CONFIG.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_CONFIG}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Always start from the same pretrained yolo11n.pt.
    #
    # Do NOT start from the best.pt of Experiment E1.
    # Otherwise E1 and E2 are no longer directly comparable.
    # --------------------------------------------------------

    model = YOLO(str(MODEL_PATH))

    results = model.train(

        # ====================================================
        # Dataset
        # ====================================================

        data=str(DATA_CONFIG),

        # ====================================================
        # Keep basic settings consistent with E1
        # ====================================================

        epochs=100,
        imgsz=960,

        # Use the same batch size as your E1 experiment.
        # If E1 used 16, keep 16 here.
        batch=32,

        device=0,
        workers=8,

        pretrained=True,
        patience=20,

        # VisDrone contains very dense scenes.
        max_det=1000,

        # ====================================================
        # Experiment E2:
        # VisDrone-oriented augmentation
        # ====================================================

        # ----- Scale augmentation -----
        # Ultralytics default is 0.5.
        # Increase moderately to expose the model to
        # a wider range of object scales.
        scale=0.7,

        # ----- Mosaic -----
        # Keep Mosaic enabled.
        # It is already enabled by default, so this is NOT
        # claimed as a newly introduced technique.
        mosaic=1.0,

        # ----- MixUp -----
        # Default is 0.
        # Introduce a small probability to improve robustness
        # without making synthetic samples dominate training.
        mixup=0.1,

        # ----- Horizontal flip -----
        # Keep the standard horizontal flip probability.
        fliplr=0.5,

        # ----- Vertical flip -----
        # Use a small probability for aerial imagery.
        # Do not make it too aggressive.
        flipud=0.1,

        # ----- Translation -----
        # Moderate positional perturbation.
        translate=0.1,

        # ----- Rotation -----
        # Aerial targets may have different orientations.
        # Keep rotation modest to avoid excessive distortion.
        degrees=5.0,

        # ----- Perspective -----
        # Keep conservative.
        perspective=0.0005,

        # ----- HSV augmentation -----
        # Mild illumination / color variation.
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # ----------------------------------------------------
        # Disable Mosaic/MixUp near the end so final epochs
        # train on more natural image distributions.
        # ----------------------------------------------------

        close_mosaic=10,

        # ====================================================
        # Output
        # ====================================================

        project=str(OUTPUT_DIR),
        name="yolo11n_visdrone_imgsz960_aug",

        exist_ok=False,

        save=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("Experiment E2 finished.")
    print("=" * 70)

    print(
        "Results saved to:\n"
        f"{OUTPUT_DIR / 'yolo11n_visdrone_imgsz960_aug'}"
    )

    return results


if __name__ == "__main__":
    main()