from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO model to static FP32 ONNX."
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "weights" / "best.pt",
        help="Path to PyTorch .pt weights.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Square input image size.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Export device, e.g. 0 or cpu.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}"
        )

    if args.model.suffix.lower() != ".pt":
        raise ValueError(
            f"Expected a .pt model, got: {args.model}"
        )

    if args.imgsz <= 0:
        raise ValueError(
            "--imgsz must be > 0"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    from ultralytics import YOLO

    print("=" * 70)
    print("Export YOLO -> ONNX")
    print("=" * 70)

    print(f"Model    : {args.model}")
    print(f"imgsz    : {args.imgsz}")
    print("batch    : 1")
    print(f"device   : {args.device}")
    print("precision: FP32")
    print("dynamic  : False")
    print("simplify : True")

    model = YOLO(str(args.model))

    exported = model.export(
        format="onnx",

        # 与当前最终模型保持一致
        imgsz=args.imgsz,

        # 后续 benchmark 也是 batch=1
        batch=1,

        # 先做固定 960x960，
        # 不增加动态 shape 变量
        dynamic=False,

        # 官方 ONNX export 支持 graph simplify
        simplify=True,

        device=args.device,
    )

    export_path = Path(str(exported))

    if not export_path.is_file():
        raise RuntimeError(
            f"Export returned {exported!r}, "
            "but the ONNX file does not exist."
        )

    print()
    print("Export successful.")
    print(f"ONNX : {export_path}")
    print(
        f"Size : "
        f"{export_path.stat().st_size / (1024 ** 2):.2f} MB"
    )


if __name__ == "__main__":
    main()