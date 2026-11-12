from pathlib import Path
from ultralytics import YOLO



PROJECT_ROOT = Path(__file__).resolve().parent.parent


MODEL_PATH = PROJECT_ROOT / "weights" / "yolo11n.pt"


IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "test.jpg"


OUTPUT_PATH = PROJECT_ROOT / "outputs"


def main():

    print(f"Model: {MODEL_PATH}")
    print(f"Image: {IMAGE_PATH}")


    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型不存在: {MODEL_PATH}"
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"测试图片不存在: {IMAGE_PATH}"
        )

    # 加载模型
    model = YOLO(str(MODEL_PATH))

    # 推理
    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device=0,
        save=True,
        project=str(OUTPUT_PATH),
        name="image_detect"
    )

    result = results[0]

    print("\n===== Detection Results =====")

    for box in result.boxes:

        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        xyxy = box.xyxy[0].tolist()

        print(
            f"class={result.names[cls_id]}, "
            f"confidence={confidence:.3f}, "
            f"bbox={xyxy}"
        )


if __name__ == "__main__":
    main()