"""Train and evaluate the SVM baseline (actor-independent split)."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import SEED, actor_split, set_seed

DATA_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("models/emotion_model.pkl")
RESULTS_DIR = Path("results")


def main():
    set_seed(SEED)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["emotion", "actor_id"])
    y = df["emotion"]
    groups = df["actor_id"].to_numpy()

    # Actor-independent split: the same test actors are used by the CNN script
    train_idx, test_idx = actor_split(groups)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("Training samples:", len(X_train), "| Testing samples:", len(X_test))
    print("Training actors:", sorted(int(a) for a in set(groups[train_idx])))
    print("Testing actors: ", sorted(int(a) for a in set(groups[test_idx])))

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale"),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Macro-F1: {macro_f1:.2%}")
    print("\nClassification Report:")
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (RESULTS_DIR / "svm_classification_report.txt").write_text(report)
    (RESULTS_DIR / "svm_metrics.json").write_text(json.dumps({
        "model": "SVM (RBF, C=10) on MFCC mean/std",
        "test_actors": sorted(int(a) for a in set(groups[test_idx])),
        "test_samples": int(len(y_test)),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
    }, indent=2))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(xticks_rotation=45)
    plt.title("SVM Emotion Recognition - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "svm_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Saved model to:", MODEL_PATH)
    print("Saved results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
