#src/stage6_visualize.py

"""
Overlay cursor bbox & event text, write mp4.
"""
import cv2, json, tqdm, glob, argparse, os
from pathlib import Path
from src.settings import OUT_DIR

def draw(f, bbox, txt):
    """Draw bounding box and text on frame with null checks"""
    if bbox:  # This handles None and empty lists automatically
        try:
            x, y, w, h = bbox
            cv2.rectangle(f, (x, y), (x+w, y+h), (0, 255, 0), 1)
        except (TypeError, ValueError):
            pass  # Skip invalid bbox formats
    
    if txt:  # Only draw text if it exists
        cv2.putText(f, str(txt), (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def run(frames_dir=str(OUT_DIR)+"/frames", 
        cursor_json="out/cursor.jsonl",
        events_json=str(OUT_DIR)+"/events.jsonl", 
        out_mp4=OUT_DIR/"annot.mp4"):
    
    # Load data with frame name checks
    cur = {}
    for line in open(cursor_json):
        data = json.loads(line)
        if "frame" not in data:
            continue
        cur[data["frame"]] = data
    
    evt = {}
    for line in open(events_json):
        data = json.loads(line)
        if "frame" not in data:
            continue
        evt[data["frame"]] = data
    
    # Get frames and verify
    frames = sorted(glob.glob(f"{frames_dir}/*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    
    # Initialize video writer
    first_frame = cv2.imread(frames[0])
    if first_frame is None:
        raise ValueError(f"Could not read first frame {frames[0]}")
    
    h, w, _ = first_frame.shape
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    
    # Process frames with error reporting
    missing_events = 0
    
    for fp in tqdm.tqdm(frames, desc="[Stage6]"):
        frame_name = Path(fp).name
        img = cv2.imread(fp)
        if img is None:
            print(f"Warning: Could not read frame {frame_name}")
            continue
        
        # Get data with fallbacks
        cursor_data = cur.get(frame_name, {"bbox": None})
        event_data = evt.get(frame_name, {"event": ""})
        
        if frame_name not in evt:
            missing_events += 1
        
        draw(img, cursor_data.get("bbox"), event_data.get("event", ""))
        vw.write(img)
    
    vw.release()
    
    # Report stats
    if missing_events > 0:
        print(f"Warning: {missing_events} frames had no event data")
    print(f"Successfully created annotated video → {out_mp4}")

if __name__=="__main__":
    run()