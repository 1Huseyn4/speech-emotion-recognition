
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

audio_files = sorted(DATA_DIR.rglob("*.wav"))
rows = []

for file in audio_files:
    parts = file.stem.split("-")

    rows.append({
        "file_path": str(file),
        "actor_id": int(parts[6]),
        "emotion_code": parts[2],
        "emotion": emotion_map[parts[2]],
    })

df = pd.DataFrame(rows)

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "metadata.csv"
df.to_csv(output_file, index=False)

print("Total audio files:", len(df))
print("\nEmotion distribution:")
print(df["emotion"].value_counts().sort_index())
print("\nSaved metadata to:", output_file)
print("\nPreview:")
print(df.head())
