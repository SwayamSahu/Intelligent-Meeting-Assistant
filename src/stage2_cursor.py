# # src/stage2_cursor.py
# """
# Detect cursor for every frame.
# If models/cursor_yolov8n.pt exists ➜ YOLO, else ➜ template match.
# Outputs out/cursor.jsonl
# """
# import cv2, json, glob, argparse, tqdm, numpy as np, os
# from pathlib import Path
# from src.settings import OUT_DIR, ROOT, YOLO_WEIGHTS, DEVICE

# TEMPLATE_DIR = ROOT / "resources" / "cursor_templates"


# def _template_detect(frame, templates, thr=0.8):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     best = None

#     for name, templ in templates:
#         print(f"Template size (before resize): {templ.shape}")

#         # Resize template using PIL
#         from PIL import Image
#         templ_pil = Image.fromarray(templ)
#         templ_resized = np.array(templ_pil.resize((100, 100), Image.LANCZOS))
#         print(f"Template size (after resize): {templ_resized.shape}")

#         res = cv2.matchTemplate(gray, templ_resized, cv2.TM_CCOEFF_NORMED)
#         _, maxVal, _, maxLoc = cv2.minMaxLoc(res)

#         if best is None or maxVal > best["score"]:
#             h, w = templ_resized.shape
#             best = {
#                 "bbox": (*maxLoc, w, h) if maxVal >= thr else None,
#                 "score": maxVal,
#                 "template": name
#             }

#     return best


# def run_template(frames_glob, out_path):
#     # Load all template images from directory
#     templates = []
#     for tp in TEMPLATE_DIR.glob("*.png"):
#         templ = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE)
#         if templ is None:
#             raise FileNotFoundError(f"Template image not found or invalid: {tp}")
#         templates.append((tp.name, templ))

#     if not templates:
#         raise FileNotFoundError(f"No templates found in {TEMPLATE_DIR}")

#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)

#     with open(out_path, "w") as f:
#         for fp in tqdm.tqdm(sorted(glob.glob(frames_glob)), desc="[Stage2-TM]"):
#             frame = cv2.imread(fp)
#             result = _template_detect(frame, templates)
#             f.write(json.dumps({
#                 "frame": os.path.basename(fp),
#                 "bbox": result["bbox"],
#                 "score": result["score"],
#                 "template": result["template"]
#             }) + "\n")


# def run_yolo(frames_glob, out_path, weights):
#     from ultralytics import YOLO
#     model = YOLO(str(weights))
#     results = model.predict(frames_glob, device=DEVICE, verbose=False)
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     with open(out_path, "w") as f:
#         for r in tqdm.tqdm(results, desc="[Stage2-YOLO]"):
#             xyxy = r.boxes.xyxy.cpu().numpy()
#             conf = r.boxes.conf.cpu().numpy()
#             bbox = None
#             if len(xyxy):
#                 x1, y1, x2, y2 = xyxy[0]
#                 bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
#             f.write(json.dumps({
#                 "frame": Path(r.path).name,
#                 "bbox": bbox,
#                 "score": float(conf[0]) if conf.size else 0
#             }) + "\n")


# def main(frames_glob=str(OUT_DIR) + "/frames/*.jpg", out_jsonl=str(OUT_DIR) + "/cursor.jsonl"):
#     if YOLO_WEIGHTS.exists():
#         run_yolo(frames_glob, out_jsonl, YOLO_WEIGHTS)
#     else:
#         run_template(frames_glob, out_jsonl)
#     print(f"Cursor detections → {out_jsonl}")


# if __name__ == "__main__":
#     main()

# src/stage2_cursor.py
"""
Detect cursor for every frame.
If models/cursor_yolov8n.pt exists ➜ YOLO, else ➜ template match.
Outputs out/cursor.jsonl
"""
import cv2, json, glob, tqdm, numpy as np, os, sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Now import settings from src
try:
    from src.settings import OUT_DIR, ROOT, YOLO_WEIGHTS, DEVICE
except ImportError:
    # If running as standalone script, define settings directly
    ROOT = Path(__file__).resolve().parent.parent
    OUT_DIR = ROOT / "out"
    YOLO_WEIGHTS = ROOT / "models" / "cursor_class_best.pt"
    DEVICE = "cpu"

TEMPLATE_DIR = ROOT / "resources" / "cursor_templates"


def _template_detect(frame, templates, thr=0.8):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best = None

    for name, templ in templates:
        # Resize template using PIL
        from PIL import Image
        templ_pil = Image.fromarray(templ)
        templ_resized = np.array(templ_pil.resize((100, 100), Image.LANCZOS))

        res = cv2.matchTemplate(gray, templ_resized, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(res)

        if best is None or maxVal > best["score"]:
            h, w = templ_resized.shape
            best = {
                "bbox": (*maxLoc, w, h) if maxVal >= thr else None,
                "score": maxVal,
                "template": name
            }

    return best


def run_template(frames_glob, out_path):
    # Load all template images from directory
    templates = []
    for tp in TEMPLATE_DIR.glob("*.png"):
        templ = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE)
        if templ is None:
            raise FileNotFoundError(f"Template image not found or invalid: {tp}")
        templates.append((tp.name, templ))

    if not templates:
        raise FileNotFoundError(f"No templates found in {TEMPLATE_DIR}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for fp in tqdm.tqdm(sorted(glob.glob(frames_glob)), desc="[Stage2-TM]"):
            frame = cv2.imread(fp)
            result = _template_detect(frame, templates)
            f.write(json.dumps({
                "frame": os.path.basename(fp),
                "bbox": result["bbox"],
                "score": result["score"],
                "template": result["template"]
            }) + "\n")


def run_yolo(frames_glob, out_path, weights):
    from ultralytics import YOLO
    import torch
    
    # Helper function to parse version strings
    def parse_version(version_str):
        parts = []
        for part in version_str.split('+')[0].split('.'):
            try:
                parts.append(int(part))
            except ValueError:
                # Handle non-numeric parts (like 'rc')
                parts.append(0)
        return parts

    torch_version = parse_version(torch.__version__)
    # Default version if parsing fails
    if not torch_version:
        torch_version = [0, 0]

    # Check for PyTorch 2.6+ (major=2, minor>=6 OR major>2)
    if (torch_version[0] == 2 and torch_version[1] >= 6) or torch_version[0] > 2:
        # Patch torch.load to force weights_only=False
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load
        try:
            model = YOLO(str(weights))
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
    else:
        # Use standard loading for older PyTorch versions
        model = YOLO(str(weights))
    
    # Rest of the function remains unchanged
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        for fp in tqdm.tqdm(sorted(glob.glob(frames_glob)), desc="[Stage2-YOLO]"):
            frame = cv2.imread(fp)
            results = model(frame, device=DEVICE, verbose=False)
            
            # Extract detection data
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            conf = results[0].boxes.conf.cpu().numpy()
            bbox = None
            
            if len(xyxy):
                # Get best detection
                best_idx = np.argmax(conf)
                x1, y1, x2, y2 = xyxy[best_idx]
                w, h = x2 - x1, y2 - y1
                bbox = (int(x1), int(y1), int(w), int(h))
                score = float(conf[best_idx])
            else:
                score = 0.0
                
            f.write(json.dumps({
                "frame": os.path.basename(fp),
                "bbox": bbox,
                "score": score
            }) + "\n")


def main(frames_glob=str(OUT_DIR) + "/frames/*.jpg", out_jsonl=str(OUT_DIR) + "/cursor.jsonl"):
    if YOLO_WEIGHTS.exists():
        run_yolo(frames_glob, out_jsonl, YOLO_WEIGHTS)
    else:
        run_template(frames_glob, out_jsonl)
    print(f"Cursor detections → {out_jsonl}")


if __name__ == "__main__":
    main()