# #src/stage3_roi_change.py

# """
# Compute SSIM change inside ROI around cursor.
# Outputs out/roi.jsonl
# """
# import json, cv2, tqdm, argparse, os
# from skimage.metrics import structural_similarity as ssim
# from pathlib import Path
# from src.settings import OUT_DIR, ROI_PAD

# def crop(img, bbox, pad):
#     x,y,w,h = bbox
#     cx, cy = x + w//2, y + h//2
#     y1, y2 = max(cy-pad,0), min(cy+pad, img.shape[0])
#     x1, x2 = max(cx-pad,0), min(cx+pad, img.shape[1])
#     return img[y1:y2, x1:x2]

# def run(frames_dir=str(OUT_DIR)+"/frames", cursor_json=str(OUT_DIR)+"/cursor.jsonl", out_json=OUT_DIR/"roi.jsonl"):
#     cursor = {j["frame"]:j for j in map(json.loads, open(cursor_json))}
#     frames = sorted(Path(frames_dir).glob("*.jpg"))
#     prev = None
#     with open(out_json,"w") as out:
#         for fp in tqdm.tqdm(frames, desc="[Stage3]"):
#             fr  = cv2.imread(str(fp))
#             bbox = cursor[fp.name]["bbox"]
#             if bbox:
#                 patch = crop(fr, bbox, ROI_PAD)
#                 score = 1.0
#                 if prev is not None and patch.size and prev.size:
#                     score,_ = ssim(
#                         cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY),
#                         cv2.cvtColor(prev , cv2.COLOR_BGR2GRAY),
#                         full=True)
#                 out.write(json.dumps({"frame":fp.name,"ssim":float(score)})+"\n")
#                 prev = patch.copy()
#     print(f"ROI SSIM → {out_json}")

# if __name__ == "__main__":
#     run()

# src/stage3_roi_change.py

import json, cv2, tqdm, argparse, os
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
from src.settings import OUT_DIR, ROI_PAD
import numpy as np

def crop(img, bbox, pad):
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    size = 2 * pad
    half = pad

    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    # Create a black square of the desired size
    patch = np.zeros((size, size, 3), dtype=np.uint8)

    # Compute where to paste the actual image content
    y1_img, y2_img = max(0, y1), min(img.shape[0], y2)
    x1_img, x2_img = max(0, x1), min(img.shape[1], x2)

    y1_patch, y2_patch = y1_img - y1, size - (y2 - y2_img)
    x1_patch, x2_patch = x1_img - x1, size - (x2 - x2_img)

    patch[y1_patch:y2_patch, x1_patch:x2_patch] = img[y1_img:y2_img, x1_img:x2_img]
    return patch

def run(frames_dir=str(OUT_DIR) + "/frames", cursor_json=str(OUT_DIR) + "/cursor.jsonl", out_json=OUT_DIR / "roi.jsonl"):
    cursor = {j["frame"]: j for j in map(json.loads, open(cursor_json))}
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    prev = None
    with open(out_json, "w") as out:
        for fp in tqdm.tqdm(frames, desc="[Stage3]"):
            fr = cv2.imread(str(fp))
            bbox = cursor[fp.name]["bbox"]
            if bbox:
                patch = crop(fr, bbox, ROI_PAD)
                score = 1.0
                if prev is not None:
                    score, _ = ssim(
                        cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
                        full=True
                    )
                out.write(json.dumps({"frame": fp.name, "ssim": float(score)}) + "\n")
                prev = patch.copy()
    print(f"ROI SSIM → {out_json}")

if __name__ == "__main__":
    run()
