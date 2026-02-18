## Cursor Event Analysis Pipeline

This project provides a multi-stage pipeline for analyzing cursor events (MOVE, HOVER, CLICK) in video recordings. The system extracts frames, detects cursor positions, tracks region-of-interest changes, classifies events, and generates both CSV reports and annotated videos.

---

#### Features

- **Frame extraction** - Extract frames from video at specified intervals  
- **Cursor detection** - Uses either YOLOv8 or template matching  
- **ROI similarity analysis** - Tracks changes around cursor position  
- **Event classification** - Rule-based or LSTM classification of cursor events  
- **Visualization** - Generates annotated videos with cursor tracking  
- **CSV reporting** - Exports event timeline to CSV format  

---

### Installation

#### Install dependencies:

`pip install -r requirements.txt`

---
### Usage
#### Run Full Pipeline
`python -m src.pipeline --video data/your_video.mp4 [--skip 5] [--device cpu]`

#### Individual Stages
#### Extract frames:
`python -m src.stage1_frames --video data/your_video.mp4 --skip 5`

#### Detect cursor:
`python -m src.stage2_cursor`

#### Compute ROI changes:
`python -m src.stage3_roi_change`

#### Classify events:
`python -m src.stage4_event_cls`

#### Export to CSV:
`python -m src.stage5_logger`

#### Generate annotated video:
`python -m src.stage6_visualize`

---

### Configuration

Modify `src/settings.py` for:

Input/output directories

Detection thresholds (SSIM_THR, VEL_MOVE, VEL_CLICK)

ROI padding size

Computation device (CPU/GPU)

---

### Project Structure

```
.
├── data/                   # Input videos
├── out/                    # Pipeline outputs
│   ├── frames/             # Extracted frames
│   ├── cursor.jsonl        # Cursor positions
│   ├── roi.jsonl           # ROI similarity scores
│   ├── events.jsonl        # Classified events
│   ├── events.csv          # Event timeline (CSV)
│   └── annot.mp4           # Annotated video
├── models/                 # Custom models
├── resources/              # Template images
├── src/
│   ├── stage1_frames.py    # Frame extraction
│   ├── stage2_cursor.py    # Cursor detection
│   ├── stage3_roi_change.py# ROI similarity
│   ├── stage4_event_cls.py # Event classification
│   ├── stage5_logger.py    # CSV export
│   ├── stage6_visualize.py # Video annotation
│   ├── settings.py         # Configuration
│   └── pipeline.py         # End-to-end pipeline
└── README.md
```
---

### Outputs

- **cursor.jsonl**: Cursor positions per frame

- **roi.jsonl**: Structural similarity scores for cursor region

- **events.jsonl**: Classified cursor events (MOVE/HOVER/CLICK)

- **events.csv**: Event timeline in CSV format

- **annot.mp4**: Video with cursor tracking and event annotations

---

### Customization

#### Cursor detection:

- Place custom YOLO model at models/cursor_class_best.pt

- Add template images to resources/cursor_templates/

#### Event classification:

- Adjust thresholds in src/settings.py:

- SSIM_THR: Similarity threshold for click detection

- VEL_MOVE: Velocity threshold for movement

- VEL_CLICK: Velocity threshold for click detection


python3 -m venv .venv
source .venv/bin/activate

pip install pyannote.audio
