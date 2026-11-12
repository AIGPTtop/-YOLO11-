from ultralytics import YOLO
import cv2
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "weights" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "datasets" / "VisDrone2019-MOT-val" / "videos" / "uav0000305_00000_v.mp4"

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "video_detect.mp4"


def main():

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )


    frame_count = 0
    total_time = 0


    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break


        start = time.time()


        results = model(
            frame,
            imgsz=640,
            conf=0.25,
            device=0
        )


        annotated_frame = results[0].plot()


        infer_time = time.time() - start
        total_time += infer_time
        frame_count += 1


        fps_now = 1 / infer_time

        cv2.putText(
            annotated_frame,
            f"FPS: {fps_now:.2f}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


        writer.write(annotated_frame)


    cap.release()
    writer.release()


    print("======================")
    print("Video detection finished")
    print(f"Frames: {frame_count}")
    print(f"Average FPS: {frame_count/total_time:.2f}")
    print(f"Saved: {OUTPUT_PATH}")
    print("======================")


if __name__ == "__main__":
    main()