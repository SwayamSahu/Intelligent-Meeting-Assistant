#src/stage1_frames.py
"""
Extract every N-th frame to out/frames.
Run:  python -m src.stage1_frames --video data/demo.mp4 --skip 2
"""
import cv2, argparse, tqdm
from pathlib import Path
from src.settings import OUT_DIR

def extract(video, skip=1, out_dir=OUT_DIR / "frames"):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = saved = 0
    with tqdm.tqdm(total=total, desc="[Stage1]") as bar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if i % skip == 0:
                cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), frame)
                saved += 1
            i += 1; bar.update(1)
    cap.release()
    print(f"Saved {saved} frames → {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--skip", type=int, default=1)
    extract(**vars(ap.parse_args()))