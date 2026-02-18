# """
# Stage 9: Speaker Diarization
# - Runs after audio transcription.
# - Adds 'speaker' column to output/audio_transcript_timestamps.csv (e.g., SPEAKER_00).
# - Uses pyannote.audio (SOTA) + timestamp overlap for assignment.
# Usage: Called automatically from pipeline.
# """
# import os
# import subprocess
# import pandas as pd
# import torch
# from pathlib import Path
# from pyannote.audio import Pipeline

# from src.settings import DEVICE  # Assumes this exists (e.g., "cuda" or "cpu" or "0")

# # HF Token from env (set via export HF_TOKEN=...)
# HF_TOKEN = os.getenv("HF_TOKEN")
# if not HF_TOKEN:
#     raise ValueError(
#         "❌ HF_TOKEN not set! "
#         "Get it from https://huggingface.co/settings/tokens and run: export HF_TOKEN=hf_..."
#     )

# def ensure_audio(video_path: str) -> str:
#     """Extract audio from video to output/audio.wav (16kHz mono) if missing."""
#     audio_path = "output/audio.wav"
#     os.makedirs("output", exist_ok=True)
#     if Path(audio_path).exists():
#         print(f"✅ Using existing audio: {audio_path}")
#         return audio_path
    
#     print(f"🔄 Extracting audio from {video_path}...")
#     cmd = [
#         "ffmpeg", "-i", video_path,
#         "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
#         audio_path, "-y"  # Overwrite
#     ]
#     try:
#         subprocess.run(cmd, check=True, capture_output=True)
#         print(f"✅ Audio extracted: {audio_path}")
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"❌ FFmpeg failed: {e.stderr.decode()}") from e
#     return audio_path

# def assign_speakers(trans_df: pd.DataFrame, diar_df: pd.DataFrame) -> pd.DataFrame:
#     """Assign speakers to transcript segments via max overlap."""
#     def get_speaker(row):
#         overlaps = []
#         for _, drow in diar_df.iterrows():
#             o_start = max(row["start"], drow["start"])
#             o_end = min(row["end"], drow["end"])
#             if o_start < o_end:
#                 overlap = o_end - o_start
#                 overlaps.append((overlap, drow["speaker"]))
#         if overlaps:
#             overlaps.sort(key=lambda x: x[0], reverse=True)  # Max overlap first
#             return overlaps[0][1]
#         return "SPEAKER_UNKNOWN"
    
#     if "speaker" in trans_df.columns:
#         print("⚠️  Transcript already diarized (skipping assignment).")
#         return trans_df
    
#     trans_df["speaker"] = trans_df.apply(get_speaker, axis=1)
#     return trans_df

# def main(video_path: str):
#     """Main entrypoint for Stage 9."""
#     print("🚀 Stage 9: Speaker Diarization")
    
#     # 1. Ensure audio
#     audio_path = ensure_audio(video_path)
    
#     # 2. Load transcript (assumes 'start', 'end' columns in seconds)
#     transcript_path = "output/audio_transcript_timestamps.csv"
#     if not Path(transcript_path).exists():
#         print("⚠️  No transcript CSV found. Run audio_stage first. Skipping.")
#         return
    
#     trans_df = pd.read_csv(transcript_path)
#     if "start" not in trans_df.columns or "end" not in trans_df.columns:
#         print(f"⚠️  Transcript missing 'start'/'end' columns. Columns: {list(trans_df.columns)}. Skipping diarization.")
#         return
    
#     # 3. Run diarization
#     print(f"🔍 Diarizing audio ({Path(audio_path).name})...")
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-community-1",
#         token=HF_TOKEN
#     )
    
#     # Device handling (matches pipeline.py)
#     if DEVICE == "cpu":
#         pipeline.to(torch.device("cpu"))
#     else:
#         pipeline.to(torch.device("cuda"))
    
#     diarization = pipeline(audio_path)
    
#     # Convert to DataFrame
#     diar_data = [
#         {"start": turn.start, "end": turn.end, "speaker": speaker}
#         for turn, _, speaker in diarization.itertracks(yield_label=True)
#     ]
#     diar_df = pd.DataFrame(diar_data)
    
#     # 4. Assign & save
#     trans_df = assign_speakers(trans_df, diar_df)
#     trans_df.to_csv(transcript_path, index=False)
    
#     print(f"✅ Diarization complete! Updated {transcript_path} with 'speaker' column.")
#     print(f"   Preview:\n{trans_df[['start', 'end', 'speaker', 'text']].head(3)}")

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         main(sys.argv[1])
#     else:
#         print("Usage: python -m src.stage9_speaker_diarization <video_path>")



"""
Stage 9: Speaker Diarization (FIXED for PyTorch unpickling & auth)
- Added safe_globals for all required classes (TorchVersion, Specifications, ListConfig)
- Explicit use_auth_token=HF_TOKEN (works with HF CLI login)
- Full MPS support for Apple Silicon
- Ignores harmless torchaudio deprecations
"""
import os
import subprocess
import shutil
import pandas as pd
import torch
import torchaudio
import warnings
from pathlib import Path
from pyannote.audio import Pipeline
import pyannote.audio.core.task  # For Specifications
import omegaconf  # For ListConfig

from src.settings import DEVICE

# Ignore torchaudio deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


torch.serialization.add_safe_globals([
    torch.torch_version.TorchVersion,
    pyannote.audio.core.task.Specifications,
    omegaconf.listconfig.ListConfig
])

def get_audio_duration(audio_path: str) -> float:
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception as e:
        print(f"⚠️ Could not read audio duration: {e}")
        return 300.0  # fallback 5 minutes

def normalize_transcript(df: pd.DataFrame, audio_path: str = None) -> pd.DataFrame:
    """Convert any timestamp column → numeric 'start' and 'end' (seconds)"""
    df = df.copy()
    
    # Detect time columns
    start_col = next((c for c in df.columns if any(k in c.lower() for k in ['start', 'begin', 'time', 'timestamp'])), None)
    end_col   = next((c for c in df.columns if any(k in c.lower() for k in ['end', 'finish', 'stop'])), None)
    
    if not start_col:
        print(f"⚠️ No time column found. Columns: {list(df.columns)}")
        return df
    
    def to_seconds(t):
        if pd.isna(t):
            return 0.0
        if isinstance(t, (int, float)):
            return float(t)
        if isinstance(t, str):
            t = t.strip()
            if ':' in t:
                try:
                    parts = [float(p) for p in t.replace(',', '.').split(':')]
                    if len(parts) == 3:   return parts[0]*3600 + parts[1]*60 + parts[2]
                    if len(parts) == 2:   return parts[0]*60 + parts[1]
                    if len(parts) == 1:   return parts[0]
                except:
                    pass
            try:
                return float(t)
            except:
                pass
        return 0.0
    
    df['start'] = df[start_col].apply(to_seconds)
    
    if end_col:
        df['end'] = df[end_col].apply(to_seconds)
    else:
        # Infer end = next start (consecutive segments)
        df = df.sort_values('start').reset_index(drop=True)
        if len(df) > 1:
            df['end'] = df['start'].shift(-1)
            # Last segment = audio end
            last_idx = df.index[-1]
            if pd.isna(df.loc[last_idx, 'end']):
                df.loc[last_idx, 'end'] = get_audio_duration(audio_path) if audio_path else df.loc[last_idx, 'start'] + 60
        else:
            df['end'] = df['start'] + 60  # single segment fallback
    
    print(f"✅ Normalized times: using start from '{start_col}', end inferred")
    return df

def ensure_audio(source_path: str) -> str:
    audio_path = "output/audio.wav"
    os.makedirs("output", exist_ok=True)
    
    if Path(audio_path).exists():
        print(f"✅ Using existing: {audio_path}")
        return audio_path
    
    ext = Path(source_path).suffix.lower()
    if ext in {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.mp4', '.mov', '.mkv'}:
        if ext in {'.wav'}:
            shutil.copy2(source_path, audio_path)
            print(f"✅ Copied audio: {source_path} → {audio_path}")
        else:
            # Extract from video
            print(f"🔄 Extracting audio from {source_path}...")
            cmd = ["ffmpeg", "-i", source_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Audio extracted → {audio_path}")
        return audio_path
    raise ValueError(f"Unsupported file: {source_path}")

def assign_speakers(trans_df: pd.DataFrame, diar_df: pd.DataFrame) -> pd.DataFrame:
    def get_speaker(row):
        overlaps = []
        for _, d in diar_df.iterrows():
            o_start = max(row["start"], d["start"])
            o_end = min(row["end"], d["end"])
            if o_start < o_end:
                overlaps.append((o_end - o_start, d["speaker"]))
        if overlaps:
            return max(overlaps, key=lambda x: x[0])[1]
        return "SPEAKER_UNKNOWN"
    
    if "speaker" in trans_df.columns:
        print("⚠️ Already has speaker column — skipping")
        return trans_df
    
    trans_df["speaker"] = trans_df.apply(get_speaker, axis=1)
    return trans_df

def main(source_path: str):
    print("🚀 Stage 9: Speaker Diarization")
    audio_path = ensure_audio(source_path)
    
    transcript_path = "output/audio_transcript_timestamps.csv"
    if not Path(transcript_path).exists():
        print("⚠️ No transcript found. Run stage7 first.")
        return
    
    trans_df = pd.read_csv(transcript_path)
    trans_df = normalize_transcript(trans_df, audio_path)
    
    if "start" not in trans_df.columns or "end" not in trans_df.columns:
        print("❌ Still no time columns after normalization")
        return
    
    # Diarization
    print(f"🔍 Diarizing {Path(audio_path).name}...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
        
    )
    
    # Device handling (CPU/MPS/CUDA)
    if str(DEVICE).lower() == "cpu":
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    pipeline.to(device)
    print(f"   Running on: {device}")
    
    diarization = pipeline(audio_path)
    diar_df = pd.DataFrame([
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ])
    
    # Assign & save
    trans_df = assign_speakers(trans_df, diar_df)
    trans_df.to_csv(transcript_path, index=False)
    
    print(f"✅ DONE! Updated {transcript_path}")
    print("Preview (first 5 rows):")
    print(trans_df[['start', 'end', 'speaker', 'text']].head().to_string(index=False))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python -m src.stage9_speaker_diarization <video_or_audio_path>")

