from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# 1. Faylların yolları
METADATA_PATH = Path("data/processed/metadata.csv")
OUTPUT_PATH = Path("data/processed/features.csv")

# 2. Metadata faylını oxu
df = pd.read_csv(METADATA_PATH)

features = []

# 3. Hər audio faylını emal et
for _, row in df.iterrows():
    signal, sr = librosa.load(row["file_path"], sr=None)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )

    # Hər MFCC üçün orta qiyməti və standart kənarlaşmanı hesabla
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    item = {}

    for i in range(13):
        item[f"mfcc_mean_{i + 1}"] = mfcc_mean[i]
        item[f"mfcc_std_{i + 1}"] = mfcc_std[i]

    item["emotion"] = row["emotion"]
    item["actor_id"] = row["actor_id"]

    features.append(item)

# 4. Nəticəni cədvələ çevir
features_df = pd.DataFrame(features)

# 5. CSV faylına yaz
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
features_df.to_csv(OUTPUT_PATH, index=False)

print("Total audio files processed:", len(features_df))
print("Feature columns:", len(features_df.columns))
print("Saved to:", OUTPUT_PATH)
print(features_df.head())