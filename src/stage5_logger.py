#src/stage5_logger.py

"""
Convert events.jsonl to CSV  (and/or SQLite later)
"""
import csv, json, argparse
from pathlib import Path
from src.settings import OUT_DIR

def to_csv(jsonl="out/events.jsonl", csv_path=OUT_DIR / "events.csv"):
    jsonl_path = Path(jsonl)
    if not jsonl_path.exists():
        print(f"❌ Error: File not found: {jsonl_path}")
        return

    lines = list(open(jsonl))
    if not lines:
        print(f"❌ Error: File is empty: {jsonl_path}")
        return

    rows = [json.loads(l) for l in lines]

    Path(csv_path).parent.mkdir(exist_ok=True, parents=True)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"✅ CSV saved → {csv_path}")

if __name__ == "__main__":
    to_csv()
