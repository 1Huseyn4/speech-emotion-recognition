"""Build the MFCC matrices used by the CNN.

Output (in data/processed/cnn_features/):
    mfcc.npy    - float32 array, shape (N, N_MFCC, MAX_LEN), zero-padded
    labels.csv  - index, emotion, actor_id, length (number of real frames)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from utils import MAX_LEN, N_MFCC, load_mfcc_matrix

METADATA_PATH = Path("data/processed/metadata.csv")
OUTPUT_DIR = Path("data/processed/cnn_features")


def main():
    df = pd.read_csv(METADATA_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matrices = []
    rows = []

    for index, row in df.iterrows():
        mfcc, length = load_mfcc_matrix(row["file_path"])
        matrices.append(mfcc)
        rows.append({
            "index": index,
            "emotion": row["emotion"],
            "actor_id": row["actor_id"],
            "length": length,
        })

    X = np.stack(matrices)
    np.save(OUTPUT_DIR / "mfcc.npy", X)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "labels.csv", index=False)

    print("Total audio files processed:", len(rows))
    print("MFCC array shape:", X.shape, f"(expected (N, {N_MFCC}, {MAX_LEN}))")
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
