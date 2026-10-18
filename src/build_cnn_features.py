
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# 1. Metadata faylını oxu
METADATA_PATH = Path("data/processed/metadata.csv")
OUTPUT_DIR = Path("data/processed/cnn_features")

df = pd.read_csv(METADATA_PATH)

# 2. MFCC parametrləri
N_MFCC = 13
MAX_LEN = 200

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

labels = []

# 3. Hər audio faylını emal et
for index, row in df.iterrows():
    signal, sr = librosa.load(
        row["file_path"],
        sr=22050
    )

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # 4. Bütün səsləri eyni uzunluğa gətir
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :MAX_LEN]

    # 5. Hər səsin MFCC matrisini saxla
    np.save(OUTPUT_DIR / f"{index}.npy", mfcc)

    labels.append({
        "index": index,
        "emotion": row["emotion"],
        "actor_id": row["actor_id"]
    })

# 6. Etiketləri ayrıca CSV faylında saxla
labels_df = pd.DataFrame(labels)
labels_df.to_csv(OUTPUT_DIR / "labels.csv", index=False)

print("Total audio files processed:", len(labels_df))
print("MFCC shape for each audio:", (N_MFCC, MAX_LEN))
print("Saved features to:", OUTPUT_DIR)
print("Saved labels to:", OUTPUT_DIR / "labels.csv")
