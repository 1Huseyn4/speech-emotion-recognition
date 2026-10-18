"""Shared helpers: constants, reproducibility, actor-independent splits, MFCC extraction."""

import random

import librosa
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------------------------------------------------------
# Constants shared by every script, so that all experiments use the same setup
# ---------------------------------------------------------------------------
SEED = 42
SAMPLE_RATE = 22050   # every audio file is resampled to this rate
N_MFCC = 13
MAX_LEN = 200         # number of MFCC frames kept for the CNN (about 4.6 s)
TEST_SIZE = 0.2       # fraction of ACTORS used for testing
VAL_SIZE = 0.15       # fraction of the remaining training ACTORS used for validation


def set_seed(seed=SEED):
    """Make Python, NumPy and PyTorch (if installed) deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def actor_split(groups, test_size=TEST_SIZE, seed=SEED):
    """Split sample indices so that no actor appears in both parts.

    The same `groups` array and the same seed always give the same split,
    so the SVM and the CNN are evaluated on exactly the same test actors.
    """
    groups = np.asarray(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), (
        "An actor appears in both parts of the split."
    )
    return train_idx, test_idx


def train_val_split(train_idx, groups, val_size=VAL_SIZE, seed=SEED):
    """Split the training indices into train/validation, again by actor."""
    groups = np.asarray(groups)
    sub_train, sub_val = actor_split(groups[train_idx], test_size=val_size, seed=seed)
    return train_idx[sub_train], train_idx[sub_val]


def load_mfcc_matrix(audio_path, n_mfcc=N_MFCC, max_len=MAX_LEN, sr=SAMPLE_RATE):
    """Load an audio file and return (mfcc, length).

    mfcc   - float32 array of shape (n_mfcc, max_len), zero-padded or cropped
    length - number of real (non-padded) frames, between 1 and max_len
    """
    signal, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

    length = min(mfcc.shape[1], max_len)
    mfcc = mfcc[:, :max_len]
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode="constant")

    return mfcc.astype(np.float32), int(length)


def frame_mask(lengths, max_len=MAX_LEN):
    """Boolean mask of shape (N, max_len): True for real frames, False for padding."""
    lengths = np.asarray(lengths)
    return np.arange(max_len)[None, :] < lengths[:, None]


def fit_normalizer(X, lengths):
    """Per-coefficient mean/std computed ONLY on real frames of the given samples.

    X       - array (N, n_mfcc, max_len)
    lengths - array (N,)
    """
    mask = frame_mask(lengths, X.shape[2])                    # (N, T)
    valid = X.transpose(0, 2, 1)[mask]                        # (n_valid_frames, n_mfcc)
    mean = valid.mean(axis=0)
    std = valid.std(axis=0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalizer(X, lengths, mean, std):
    """Standardise real frames and keep padded frames at exactly zero."""
    X = (X - mean[None, :, None]) / std[None, :, None]
    mask = frame_mask(lengths, X.shape[2])[:, None, :]        # (N, 1, T)
    return (X * mask).astype(np.float32)
