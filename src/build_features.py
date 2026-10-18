"""Build the tabular MFCC features (mean + std per coefficient) used by the SVM."""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from utils import N_MFCC, SAMPLE_RATE

METADATA_PATH = Path("data/processed/metadata.csv")
OUTPUT_PATH = Path("data/processed/features.csv")


def main():
    df = pd.read_csv(METADATA_PATH)
    features = []

    for _, row in df.iterrows():
        signal, sr = librosa.load(row["file_path"], sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        item = {}
        for i in range(N_MFCC):
            item[f"mfcc_mean_{i + 1}"] = mfcc_mean[i]
            item[f"mfcc_std_{i + 1}"] = mfcc_std[i]

        item["emotion"] = row["emotion"]
        item["actor_id"] = row["actor_id"]
        features.append(item)

    features_df = pd.DataFrame(features)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)

    print("Total audio files processed:", len(features_df))
    print("Feature columns:", len(features_df.columns))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
