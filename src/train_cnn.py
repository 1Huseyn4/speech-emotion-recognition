"""Train and evaluate the CNN (actor-independent train / validation / test split).

Protocol
--------
* Test set      : the SAME held-out actors as the SVM (see utils.actor_split).
* Validation set: a few of the remaining actors, used for early stopping only.
* Normalisation : per-MFCC-coefficient mean/std computed on training data only.
* Augmentation  : random time and frequency masking on training batches.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from model import EmotionCNN
from utils import (
    MAX_LEN,
    N_MFCC,
    SEED,
    actor_split,
    apply_normalizer,
    fit_normalizer,
    set_seed,
    train_val_split,
)

DATA_DIR = Path("data/processed/cnn_features")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

BATCH_SIZE = 32
MAX_EPOCHS = 80
PATIENCE = 12          # stop if validation loss does not improve for this many epochs
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def augment(batch):
    """SpecAugment-style masking applied independently to each sample in a batch."""
    batch = batch.clone()
    for i in range(batch.shape[0]):
        t_width = np.random.randint(0, 21)
        t_start = np.random.randint(0, MAX_LEN - t_width + 1)
        batch[i, :, :, t_start:t_start + t_width] = 0.0

        f_width = np.random.randint(0, 3)
        f_start = np.random.randint(0, N_MFCC - f_width + 1)
        batch[i, :, f_start:f_start + f_width, :] = 0.0
    return batch


def evaluate_loss(model, X, y, criterion):
    model.eval()
    with torch.no_grad():
        return criterion(model(X), y).item()


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---------------------------------------------------------------- data
    labels_df = pd.read_csv(DATA_DIR / "labels.csv")
    X = np.load(DATA_DIR / "mfcc.npy")                       # (N, N_MFCC, MAX_LEN)
    lengths = labels_df["length"].to_numpy()
    groups = labels_df["actor_id"].to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels_df["emotion"])
    classes = label_encoder.classes_.tolist()

    train_val_idx, test_idx = actor_split(groups)
    train_idx, val_idx = train_val_split(train_val_idx, groups)

    print("Samples  train/val/test:", len(train_idx), len(val_idx), len(test_idx))
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        print(f"Actors   {name:<5}:", sorted(int(a) for a in set(groups[idx])))

    # Normalisation statistics come from the training samples only
    mean, std = fit_normalizer(X[train_idx], lengths[train_idx])

    def prepare(idx):
        Xn = apply_normalizer(X[idx], lengths[idx], mean, std)[:, np.newaxis, :, :]
        return (
            torch.tensor(Xn, dtype=torch.float32).to(device),
            torch.tensor(y[idx], dtype=torch.long).to(device),
        )

    X_train, y_train = prepare(train_idx)
    X_val, y_val = prepare(val_idx)
    X_test, y_test = prepare(test_idx)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # --------------------------------------------------------------- model
    model = EmotionCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------ training
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(augment(X_batch)), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate_loss(model, X_val, y_val, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch [{epoch:02d}/{MAX_EPOCHS}]  train loss: {train_loss:.4f}  val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val loss {best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)

    # ---------------------------------------------------------- evaluation
    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(X_test), dim=1)

    y_true = y_test.cpu().numpy()
    y_pred = predictions.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0,
    )

    print(f"\nTest Accuracy: {accuracy:.2%}")
    print(f"Test Macro-F1: {macro_f1:.2%}")
    print("\nClassification Report:")
    print(report)

    # ------------------------------------------------------------- outputs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    (RESULTS_DIR / "cnn_classification_report.txt").write_text(report)
    (RESULTS_DIR / "cnn_metrics.json").write_text(json.dumps({
        "model": "CNN on 13x200 MFCC matrices",
        "test_actors": sorted(int(a) for a in set(groups[test_idx])),
        "test_samples": int(len(y_true)),
        "epochs_trained": len(history["train_loss"]),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
    }, indent=2))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
        display_labels=classes,
        xticks_rotation=45,
        cmap="Blues",
    )
    plt.title("CNN Emotion Recognition - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cnn_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cnn_training_curve.png", dpi=200)
    plt.close()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "emotion_classes": classes,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "n_mfcc": N_MFCC,
            "max_len": MAX_LEN,
        },
        MODEL_DIR / "emotion_cnn.pth",
    )

    print("Saved model to:", MODEL_DIR / "emotion_cnn.pth")
    print("Saved results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
