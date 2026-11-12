from pathlib import Path
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = PROJECT_ROOT / "datasets" / "VisDrone2019-MOT-val"

IMG_ROOT = DATASET / "sequences"
OUT_ROOT = DATASET / "videos"

OUT_ROOT.mkdir(exist_ok=True)


def images_to_video(img_dir, out_file, fps=25):

    imgs = sorted(img_dir.glob("*.jpg"))

    if len(imgs) == 0:
        return

    first = cv2.imread(str(imgs[0]))

    h, w = first.shape[:2]

    writer = cv2.VideoWriter(
        str(out_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    for img in imgs:
        frame = cv2.imread(str(img))

        if frame is not None:
            writer.write(frame)

    writer.release()


for seq in IMG_ROOT.iterdir():

    if seq.is_dir():

        print("processing:", seq.name)

        out = OUT_ROOT / f"{seq.name}.mp4"

        images_to_video(
            seq,
            out
        )

print("done")