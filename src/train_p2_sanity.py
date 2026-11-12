from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_CONFIG = PROJECT_ROOT / "configs" / "yolo11n_p2.yaml"
PRETRAINED = PROJECT_ROOT / "weights" / "yolo11n.pt"
DATA_CONFIG = PROJECT_ROOT / "configs" / "visdrone.yaml"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "train"


def main():

    print("=" * 70)
    print("YOLO11n-P2 Sanity Check")
    print("=" * 70)

    print(f"Model config : {MODEL_CONFIG}")
    print(f"Pretrained   : {PRETRAINED}")
    print(f"Dataset      : {DATA_CONFIG}")

    # --------------------------------------------------------
    # 1. Build the modified architecture
    # --------------------------------------------------------

    model = YOLO(str(MODEL_CONFIG))

    # --------------------------------------------------------
    # 2. Transfer compatible pretrained YOLO11n weights
    #
    # The new P2-related layers have no corresponding weights
    # in the original model and remain newly initialized.
    # --------------------------------------------------------

    model.load(str(PRETRAINED))

    print("\nPretrained compatible weights loaded.\n")

    model.info()

    # --------------------------------------------------------
    # 3. Only 1 epoch.
    # This is NOT the formal experiment.
    # --------------------------------------------------------

    model.train(
        data=str(DATA_CONFIG),

        epochs=1,
        imgsz=960,

        # Must eventually match E1.
        batch=16,

        device=0,
        workers=8,

        # Keep E1/default augmentation settings.
        # Do NOT use the E2 enhanced augmentation here.

        max_det=1000,

        project=str(OUTPUT_DIR),
        name="yolo11n_p2_sanity",

        exist_ok=True,

        save=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()