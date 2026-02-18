#src/settings.py

from pathlib import Path

### Paths ###
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "out"
MODEL_DIR= ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

### Cursor detection ###
YOLO_WEIGHTS = MODEL_DIR / "cursor_class_best.pt"   # leave empty → template matching
TEMPLATE     = ROOT / "resources" / "arrow_template.png"

### ROI / SSIM ###
ROI_PAD   = 50
SSIM_THR  = 0.90

### Event thresholds (pixels per frame) ###

VEL_MOVE  = 10
VEL_CLICK = 2

### Device ###
DEVICE = "cpu"    # "0" for GPU 0,  "cpu" to force CPU