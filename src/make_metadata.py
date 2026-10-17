from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

rows = []
for path in sorted(DATA_DIR.glob("Actor_*/*.wav")):
    parts = path.stem.split("-")  # ['03','01','05','01','02','01','12']
    rows.append({
        "path": path.as_posix(),
        "emotion": EMOTIONS[parts[2]],
        "intensity": "normal" if parts[3] == "01" else "strong",
        "actor": int(parts[6]),
    })

df = pd.DataFrame(rows)
df.to_csv("metadata.csv", index=False)

print("Cədvəlin ölçüsü:", df.shape)
print(df.head())
print("\nHər emosiyada nümunə sayı:")
print(df["emotion"].value_counts())
print("\nAktyor sayı:", df["actor"].nunique())