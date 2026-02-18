#src/pipeline.py

"""
Single command to run stages 1→6.
Usage:
  python -m src.pipeline --video data/demo.mp4
  python -m src.pipeline --video data/demo.mp4 --skip 2 --device 0
Optional flags:
  --skip 5          # keep every 5th frame
  --device cpu      # override default GPU
"""
import argparse, sys, os
from src.stage1_frames   import extract
from src.stage2_cursor   import main as cursor_det
from src.stage3_roi_change import run as roi_change
from src.stage4_event_cls  import main as evt_cls
from src.stage5_logger   import to_csv
from src.stage6_visualize  import run as visualize
from src.settings import DEVICE
from src.stage7_audio_transcribe_intent_summarize     import main as audio_stage
from src.stage8_gifs       import create_gif_report 

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",required=True)
    ap.add_argument("--skip",type=int,default=1)
    ap.add_argument("--device",default=DEVICE)
    ap.add_argument("--gif_timestamps", nargs="+", default=["00:00:30", "00:01:00.500", 150], 
                    help="Timestamps for GIF report")
    args=ap.parse_args()

    os.environ["YOLO_VERBOSE"]="False"
    if args.device!="":
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device if args.device!="cpu" else "-1"

    extract(args.video,args.skip)
    cursor_det()
    roi_change()
    evt_cls()
    to_csv()
    visualize()
    audio_stage(args.video)
    create_gif_report(args.video, xlsx_path="output/output.xlsx", csv_path="out/events.csv", transcript_csv="output/audio_transcript_timestamps.csv")
    print("\n✅  Pipeline finished — check the out/ folder and output/ folders")

if __name__=="__main__":
    # import sys
    # sys.argv =[
    #     "pipeline.py",
    #     "--video",r"C:\Users\Downloads\PythonProject2\PythonProject2_enhanced\data\windows_detect.mp4",
    #     "--skip", "2",
    #     "--device","0"
    #     ]
    main()

