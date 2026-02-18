# pip install openpyxl-3.1.5
# pip install transformers==4.40.2 datasets==2.18.0
# pip install transformers[torch]
#pip install llama-cpp-python accelerate
#pip install transformers datasets scikit-learn pandas
#xcopy C:\Users\SashikantaMishra\PycharmProjects\PythonProject2\models\pyannote_speaker_diarization
#The  current active token is: `pyannote-diarization`
#pip  install imageio-ffmpeg openai-whisper torch tqdm
#run  huggingface-cli login
#run  HF_HUB_DISABLE_SYMLINKS_WARNING true


#         Excel File
#               ↓
#       [ Preprocessing ]
#                ↓
#   Chunked & Tokenized Text → HF Dataset
#                 ↓
#     Fine-tune BART on binary intent
#                 ↓
#           Trained Model
#                 ↓
#       predict("Let me share...")
#                 ↓
#    Output: {"prediction": "screen_share_intent", "score": 0.98}


# ==============================================================================
# 📌 Video → Insights Pipeline
# ==============================================================================
# This script builds an end-to-end pipeline for extracting insights from videos.
#
# 🔹 Features Implemented:
#   1. **Audio Extraction** → Extracts audio from video using FFmpeg
#   2. **Speech Transcription** → Converts audio to text with Whisper
#   3. **Speaker Diarization** → Identifies who spoke when (PyAnnote)
#   4. **Intent Classification** → Fine-tunes BART model on Excel dataset
#   5. **Report Generation** → Saves "screen_share_intent" predictions to Excel
#   6. **Summarization** → Summarizes transcript using Mistral (via Ollama)
#   7. **Frame Capture** → Extracts video frames at fixed intervals
#
# ==============================================================================
# 📊 FLOW OF THE PIPELINE
# ==============================================================================
#
#   ┌────────────┐
#   │  Video.mp4 │
#   └──────┬─────┘
#          │
#          ▼
#   ┌───────────────┐
#   │ Audio Extract │───────────────┐
#   └──────┬────────┘               │
#          │                        │
#          ▼                        │
#   ┌───────────────┐   ┌───────────────────┐
#   │ Transcription │ → │ Transcript.txt    │
#   └──────┬────────┘   │ Transcript.csv    │
#          │            └───────────────────┘
#          ▼
#   ┌───────────────────┐
#   │ Intent Classifier │ → output.xlsx
#   └──────┬────────────┘
#          │
#          ▼
#   ┌───────────────────┐
#   │ Summarization     │ → summary.txt
#   └──────┬────────────┘
#          │
#          ▼
#   ┌───────────────────┐
#   │ Frame Capture     │ → /out_frames/
#   └───────────────────┘
#
# =======================================================================================


#!/usr/bin/env python3

import os,warnings

from pyarrow import duration

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
import cv2
import sys
import whisper
import transformers
print(transformers.__version__)
import torch
import numpy as np
import subprocess
import argparse
import platform
import wave
from pyannote.audio import Pipeline,Model
from pydub import AudioSegment
import argparse
from huggingface_hub import snapshot_download
import json
import time
import requests
from sklearn.model_selection import train_test_split
#from pyannote.audio.pipelines.utils import  load_pretrained_pipeline
from datetime import timedelta
import csv
import subprocess
import json
import pandas as pd

# ─── 0) Silence the FP16 warning Whisper throws on CPU ────────────────────────
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead"
)
#─── 0.5) Special Bundle for Intent Oriented Model ──────────────────────── 
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import torch

# ────────────────────────────────
# GLOBAL CONSTANTS
# ────────────────────────────────

# Ensure output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
MODEL_NAME = "facebook/bart-large-mnli"
EXCEL_PATH = "input/Intent_Data.xlsx"  # Input intent dataset
# EXCEL_PATH = "data.xlsx"
TEXT_COLUMN = "transcript"
LABEL_COLUMN = "intent"
LABEL_MAPPING = {"screen_share_intent": 1, "no_intent": 0}
MAX_WORDS = 80   # Chunk size
OVERLAP = 10     # Word overlap between chunks


# ──────────────────────────────────────────────────────────────────
# 0. HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────

def split_transcript(text, max_words=MAX_WORDS, overlap=OVERLAP):
    """Split transcript text into overlapping word chunks for context."""
    words = str(text).split()  # Ensure text is string
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += max_words - overlap
    return chunks

# ──────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS DATASET
# ──────────────────────────────────────────────────────────────────

# Step 1: Load and preprocess the dataset
def load_dataset(path=EXCEL_PATH):
    """Load Excel intent dataset, clean it, and split into train/test."""
    df = pd.read_excel(path, engine="openpyxl")
    df = df.rename(columns={TEXT_COLUMN: "text", LABEL_COLUMN: "label"})
 ## STEP 2: Preprocessing
    df["label"] = df["label"].map(LABEL_MAPPING)

    # Drop invalid rows
    df = df.dropna(subset=["text", "label"])

    #STEP 3 Convert the "intent" column into 0 or 1 using LABEL_MAPPING
    df = df[df["label"].isin([0, 1])]

    #STEP 4 Chunking
    chunked_rows = []
    for _, row in df.iterrows():
# Chunk transcripts (e.g., 300 words)into chunks of up to 80 words, with 10-word overlaps for context.
        chunks = split_transcript(row["text"])
        for chunk in chunks:
            chunked_rows.append({
                "text": chunk,
                "label": int(row["label"])
            })

    # Build final DataFrame from Chunk Rows
    chunked_df = pd.DataFrame(chunked_rows)

    # Stratified split into train/test sets
    train_df, test_df = train_test_split(
        chunked_df,
        test_size=0.2,
        stratify=chunked_df["label"],
        random_state=42
    )

    # Convert to HuggingFace datasets
    dataset = {
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True))
    }

    return dataset

# Step 2: Tokenization
def tokenize_function(example):
    """Tokenize input text for BART model."""
    return tokenizer(example["text"], truncation=True)

# Step 3: Evaluation metrics
def compute_metrics(eval_pred):
    """Custom evaluation metrics for classification."""
    print("Start Computing metrics...")
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # <-- Fix for your error
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)

    reverse_labels = {v: k for k, v in LABEL_MAPPING.items()}  #()}
    label_names = [reverse_labels[i] for i in sorted(reverse_labels.keys())]


    report = classification_report(
        labels,
        preds,
        target_names=label_names,
        labels=[0, 1],  # explicitly tell sklearn what to expect
        output_dict=True,
        zero_division=0  # avoid divide-by-zero error when a class is missing
    )
    print("Finished Computing metrics...")
    return {
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }
# ==================================================================================
# chunk of text is tokenized (converted to input IDs + attention masks)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True
)

# Load and tokenize dataset
dataset = load_dataset()
tokenized_dataset = {
    "train": dataset["train"].map(tokenize_function, batched=True),
    "test": dataset["test"].map(tokenize_function, batched=True)
}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./output/results",
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./output/logs",
    logging_steps=10,
    
    use_cpu=True,   

    # load_best_model_at_end=True,
    # metric_for_best_model="f1",
    save_total_limit=1,
    save_safetensors=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Step 4: Train
trainer.train()

# Saving the trainer and tokenizer to avoid crashing

trainer.save_model("trained_bart_model")
tokenizer.save_pretrained("trained_bart_model")
MODEL_PATH = "trained_bart_model"

# ─────────────────────────────────────────────────────────────
# 2. AUDIO EXTRACTION
# ─────────────────────────────────────────────────────────────

import imageio_ffmpeg as iio_ffmpeg

FFMPEG_BIN = iio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_BIN)
print("Using ffmpeg at:", FFMPEG_BIN)
AudioSegment.converter = FFMPEG_BIN


def extract_audio(video_path: str, audio_path: str = os.path.join(OUTPUT_DIR, "audio.wav")) -> str:
    """
    Use the bundled ffmpeg.exe to extract a mono, 16 kHz WAV from the video.

    Extracts mono 16kHz WAV audio from video using FFmpeg.
    """
    print("Started extract_audio")
    cmd = [
        FFMPEG_BIN,
        "-i", video_path,
        "-vn",                   # drop video track
        "-acodec", "pcm_s16le",  # raw PCM 16-bit little-endian
        "-ac", "1",              # mono
        "-ar", "16000",          # 16 kHz
        audio_path,
        "-y"                     # overwrite if exists
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"[ffmpeg not found] tried to run {FFMPEG_BIN}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ffmpeg error] {e.stderr}", file=sys.stderr)
        sys.exit(1)
    print("Exited extract_audio")
    return audio_path


# ────────────────────────────────────────────────────────────────   
# 3. TRANSCRIPTION
# ──────────────────────────────────────────────────────────────── 

def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))


def transcribe_local(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribes audio to text and writes timestamped segments every 5s to a CSV.
    Returns the full transcript as a string.
    """
    print("Started transcribe_local")
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        print(f"[whisper load error] {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with wave.open(audio_path, "rb") as wf:
            sr = wf.getframerate()
            if sr != 16000:
                print(f"[warning] WAV is {sr} Hz, expected 16000 Hz", file=sys.stderr)
            frames = wf.readframes(wf.getnframes())
            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        print(f"[audio load error] {e}", file=sys.stderr)
        sys.exit(1)

    try:
        audio_tensor = torch.from_numpy(audio_np)
        result = model.transcribe(audio_tensor)
    except Exception as e:
        print(f"[transcription error] {e}", file=sys.stderr)
        sys.exit(1)

    # Get full transcript
    full_transcript = result.get("text", "")

    # Save full transcript as transcript.txt inside output/
    transcript_path = os.path.join(OUTPUT_DIR, "transcript.txt")
    try:
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(full_transcript)
        print(f"[output] Transcript written to: {transcript_path}")
    except Exception as e:
        print(f"[transcript write error] {e}", file=sys.stderr)

    # Process segments for timestamped CSV output
    segments = result.get("segments", [])
    bin_size = 5  # seconds
    csv_rows = []
    current_bin_start = 0
    current_text = ""

    for segment in segments:
        seg_start = segment["start"]
        seg_text = segment["text"].strip()

        # If segment exceeds current 30s bin, flush and start new
        while seg_start >= current_bin_start + bin_size:
            if current_text:
                csv_rows.append([format_time(current_bin_start), current_text.strip()])
            current_bin_start += bin_size
            current_text = ""

        current_text += " " + seg_text

    # Add any remaining text
    if current_text:
        csv_rows.append([format_time(current_bin_start), current_text.strip()])

    # Write to CSV
    # csv_path = os.path.splitext(audio_path)[0] + "_transcript_timestamps.csv"

    # Write to CSV inside output/
    csv_path = os.path.join(OUTPUT_DIR, "audio_transcript_timestamps.csv")
    print(csv_path)
    try:
        with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_start", "text"])
            writer.writerows(csv_rows)
        print(f"[output] CSV written to: {csv_path}")
    except Exception as e:
        print(f"[csv write error] {e}", file=sys.stderr)
    
    print("Exited transcribe_local")
    return full_transcript

# ──────────────────────────────────────────────────────────────────
# 4. INTENT CLASSIFICATION → Excel
# ──────────────────────────────────────────────────────────────────

# Function to generate Excel from Transcript CSV ────────────────────────────────────────────

def generate_excel_from_transcript(csv_path: str, output_excel_path: str):
    """
    Reads a timestamped transcript CSV file and generates an Excel file
    with rows predicted as 'screen_share_intent'.

    Generate Excel report containing only 'screen_share_intent' predictions.
    """
    import pandas as pd
    import torch

    df = pd.read_csv(csv_path)  # must have columns: timestamp_start, text
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    # Combine short sentences and group intelligently
    grouped_texts = []
    temp_text = ""
    temp_timestamp = ""

    for idx, row in df.iterrows():
        text = row["text"].strip()
        timestamp = row["timestamp_start"]

        if not temp_text:
            temp_timestamp = timestamp

        temp_text += " " + text

        if any(temp_text.strip().endswith(p) for p in [".", "!", "?"]) or len(temp_text.split()) > 15:
            grouped_texts.append({
                "timestamp_start": temp_timestamp,
                "text": temp_text.strip()
            })
            temp_text = ""
            temp_timestamp = ""

    # Function to return predicted label
    def predict_label(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_class = outputs.logits.argmax().item()
        return {v: k for k, v in LABEL_MAPPING.items()}[predicted_class]

    # Apply prediction to grouped texts
    results = []
    for row in grouped_texts:
        label = predict_label(row["text"])
        if label == "screen_share_intent":
            results.append({
                "timestamp_start": row["timestamp_start"],
                "text": row["text"],
                "intent": label
            })

    # Save to Excel
    output_df = pd.DataFrame(results)
    output_df.to_excel(output_excel_path, index=False)
    print(f"✅ Output written to {output_excel_path}")

# ─── 1) Bundle a private ffmpeg.exe via imageio-ffmpeg ────────────────────────


# ─── 2) Monkey-patch ctypes.find_library BEFORE importing whisper ─────────────
if platform.system() == "Windows":
    import ctypes.util
    _orig_find = ctypes.util.find_library
    ctypes.util.find_library = (
        lambda name: "msvcrt" if name == "c" else _orig_find(name)
    )

# ───────────────────────────────────────────────────────────────────
# 5. SUMMARIZATION (Mistral/Ollama)
# ───────────────────────────────────────────────────────────────────

# ─── Summarization with Mistral ────────────────────────────────────────────────────

def summarize_with_mistral(transcript_path: str, output_path: str, model: str = "llama3.2"):
    """
    Uses Ollama's mistral:7b model to summarize transcript and saves output in a txt file.
    """
    # Read transcript text
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    # Create summarization prompt
    prompt = f"Summarize the following meeting transcript into clear and concise bullet points:\n\n{transcript}"

    # Run Ollama mistral model
    cmd = ["ollama", "run", model, prompt]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8"
    )
    summary, err = process.communicate()

    if err:
        print("⚠️ Ollama Error:", err)

    if not summary.strip():
        print("⚠️ No summary generated.")
        summary = "Summary could not be generated."    

    # Save summary
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary.strip())

    print(f"✅ Summary saved to {output_path}")
    return summary.strip()

# ─────────────────────────────────────────────────────────────────
# 6. VIDEO FRAME CAPTURE
# ─────────────────────────────────────────────────────────────────
# ========= Screen Capture =======

def capture_frame_at_time(cap, timestamp, output_path):
    """
    Seeks to `timestamp` seconds in `cap` VideoCapture, grabs one frame, and writes it.
    """
    print("Inside capture_frame_at_time")
    fps      = cap.get(cv2.CAP_PROP_FPS)

    frame_no = int(fps * timestamp)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration=frame_count / fps

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read frame at {timestamp}s (frame {frame_no})")

    cv2.imwrite(output_path, frame)    
    print("Exited capture_frame_at_time")

def main(video_path=None):
    print("Started Main")

    if video_path is None:

        parser = argparse.ArgumentParser(
            description="Transcribe a video file via local Whisper with TimeStamps"
        )
        parser.add_argument(
            "video",
            nargs="?",
            default="input/cursor_tracking.mp4",
            help=(
                "Path to input video file\n"
                "If not provided, the default video will be used:\n"
                "input\\cursor_tracking.mp4"
            )
        )
        parser.add_argument(
            "--model",
            default="base",
            choices=["tiny", "base", "small", "medium", "large"],
            help="Whisper model size to use"
        )
        args = parser.parse_args()
        video_path = args.video
        model_size = args.model
    else:
        # Pipeline mode
        model_size = "base"    

    # ── Step 1: Extract audio from video ──────────────────────────────────────────────
    audio_file = extract_audio(video_path)

    # ── Step 2: Transcription ──────────────────────────────────────────────
    transcript = transcribe_local(audio_file, model_size)
    transcript_path = "transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"\nTranscript written to {transcript_path}\n")
    
    # ── Step 3: Generate Excel from Transcript ──────────────────────────────    
    generate_excel_from_transcript(
    csv_path=os.path.join("output", "audio_transcript_timestamps.csv"),
    output_excel_path=os.path.join("output", "output.xlsx")
    )
    
    # ── Step 4: Intelligent summary with Mistral ─────────────────────────────
    summary_path = os.path.join("output", "summary.txt")
    summarize_with_mistral(transcript_path, summary_path, model="llama3.2") 

##================================================================================
    # VIDEO_PATH = args.video
    OUTPUT_DIR = "output/out_frames"
    START_TIME_SEC = 0.0
    GAP_SEC = 5
## ===============   Screen Capture   =======================================

    # Capture Frames
    # ── Step 5: Capture Frames ─────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    # Get video duration in seconds
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0

    ITERATIONS = int(duration // GAP_SEC)

    for i in range(ITERATIONS):
        # compute timestamp for this iteration
        t = START_TIME_SEC + i * GAP_SEC
        filename = os.path.join(OUTPUT_DIR, f"shot_{i + 1:02d}_{int(t)}s.png")

        try:
            capture_frame_at_time(cap, t, filename)
            print(f"[{i + 1}/{ITERATIONS}] Saved frame at {t}s → {filename}")
        except Exception as e:
            print(f"[{i + 1}/{ITERATIONS}] ERROR grabbing at {t}s: {e}")

        # wait before the next capture (real-time delay)
        if i < ITERATIONS - 1:
            time.sleep(GAP_SEC)

    cap.release()
    print("Done.")


if __name__ == "__main__":
    print("Stage-7 started")
    main()
