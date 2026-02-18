#src/stage4_event_cls.py
"""
Rule-based MOVE / HOVER / CLICK.
If models/lstm.pt exists ➜ use that instead.
"""
import json, math, tqdm, torch, argparse, os
from pathlib import Path
from src.settings import OUT_DIR, SSIM_THR, VEL_MOVE, VEL_CLICK, MODEL_DIR, DEVICE

def dist(b1, b2):
    if not b1 or not b2: return 1e9
    return math.hypot(
        (b1[0] + b1[2] / 2) - (b2[0] + b2[2] / 2),
        (b1[1] + b1[3] / 2) - (b2[1] + b2[3] / 2)
    )

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

def rule(cursor_json, roi_json, out_json, fps=30):
    cur = {j["frame"]: j for j in map(json.loads, open(cursor_json))}
    roi = {j["frame"]: j for j in map(json.loads, open(roi_json))}
    frames = sorted(cur)
    last = None
    events = []
    skipped = 0

    for f in frames:
        if f not in roi:
            print(f"⚠️ Skipping frame {f} – no ROI data found.")
            skipped += 1
            continue

        bbox = cur[f]["bbox"]
        ssim = roi[f]["ssim"]
        v = dist(bbox, last)

        if v < VEL_CLICK and ssim < SSIM_THR:
            evt = "CLICK"
        elif v < VEL_MOVE:
            evt = "HOVER"
        else:
            evt = "MOVE"

        # Extract frame number from filename like "000062.jpg"
        frame_num = int(f.replace(".jpg", ""))
        timestamp = format_time(frame_num / fps)    

        events.append({"frame": f, "frame_timestamps": timestamp, "event": evt, "vel": v, "ssim": ssim})
        last = bbox

    with open(out_json, "w") as w:
        for e in events:
            w.write(json.dumps(e) + "\n")

    print(f"\n✅ Events → {out_json}")
    print(f"ℹ️  Processed {len(frames) - skipped} frames, ❌ Skipped {skipped} due to missing ROI")

# placeholder for future LSTM
def lstm_classify(*_):
    print("LSTM model not implemented in this snippet.")

def main(cursor="out/cursor.jsonl", roi="out/roi.jsonl", out_json=OUT_DIR/"events.jsonl", fps=30):
    ckpt = MODEL_DIR / "lstm.pt"
    if ckpt.exists():
        lstm_classify(cursor, roi, out_json)
    else:
        rule(cursor, roi, out_json, fps=fps)

if __name__ == "__main__":
    main()
