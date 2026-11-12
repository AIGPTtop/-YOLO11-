from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_CONFIG = PROJECT_ROOT / "configs" / "yolo11n_p2.yaml"


def main():

    print("=" * 70)
    print("YOLO11n-P2 Structure Check")
    print("=" * 70)

    print(f"Model config: {MODEL_CONFIG}")

    if not MODEL_CONFIG.exists():
        raise FileNotFoundError(
            f"Config not found: {MODEL_CONFIG}"
        )

    # Build model from YAML only.
    # No training here.
    model = YOLO(str(MODEL_CONFIG))

    print("\nModel created successfully.\n")

    model.info(detailed=True)

    print("\n" + "=" * 70)
    print("P2 model structure check passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()