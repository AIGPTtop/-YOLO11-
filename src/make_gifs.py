"""Build README demo GIFs from the existing tracking / detection videos."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

WIDTH = 560
STEP = 3
DELAY_MS = 80


def render(src: Path, dst: Path, caption: str) -> None:
    cap = cv2.VideoCapture(str(src))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % STEP == 0:
            h, w = frame.shape[:2]
            scale = WIDTH / w
            frame = cv2.resize(frame, (WIDTH, int(round(h * scale))), interpolation=cv2.INTER_AREA)
            bar = np.zeros((34, frame.shape[1], 3), dtype=np.uint8)
            frame = np.vstack([bar, frame])
            cv2.putText(frame, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {src}")

    base = Image.fromarray(frames[0]).convert("RGB").quantize(colors=64, method=Image.MEDIANCUT)
    out = [Image.fromarray(f).convert("RGB").quantize(palette=base, dither=Image.FLOYDSTEINBERG) for f in frames]
    out[0].save(dst, save_all=True, append_images=out[1:], duration=DELAY_MS, loop=0, optimize=True)
    size_mb = dst.stat().st_size / 1e6
    print(f"{dst.name:22s} frames={len(frames):3d}  {size_mb:5.2f} MB")


JOBS = [
    ("outputs/video_detect.mp4", "detect.gif", "YOLO11n-P2-960  ·  detection"),
    ("outputs/track_bytetrack/uav0000305_00000_v.avi", "bytetrack.gif", "ByteTrack  ·  no camera-motion compensation"),
    ("outputs/track_botsort_gmc/uav0000305_00000_v.avi", "botsort_gmc.gif", "BoT-SORT + sparseOptFlow GMC"),
]

for src, name, caption in JOBS:
    render(ROOT / src, ASSETS / name, caption)
