
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# 1. Fayl yolları
DATA_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("models/emotion_model.pkl")


def main():
    # 2. Xüsusiyyətləri oxu
    df = pd.read_csv(DATA_PATH)

    # 3. Xüsusiyyətləri (X), etiketləri (y) və aktyor qruplarını ayır
    X = df.drop(columns=["emotion", "actor_id"])
    y = df["emotion"]
    groups = df["actor_id"]

    # 4. Dataset-i aktyorlara görə train və test hissələrinə böl
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_actors = set(groups.iloc[train_idx])
    test_actors = set(groups.iloc[test_idx])

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training actors:", sorted(train_actors))
    print("Testing actors:", sorted(test_actors))

    # 5. Eyni aktyorun hər iki dəstə düşmədiyini yoxla
    assert train_actors.isdisjoint(test_actors), (
        "Error: An actor appears in both training and testing sets."
    )

    # 6. SVM modelini yarat
    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
        ),
    )

    # 7. Modeli öyrət
    model.fit(X_train, y_train)

    # 8. Test dəsti üçün proqnoz ver
    y_pred = model.predict(X_test)

    # 9. Nəticələri qiymətləndir
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    # 10. Confusion matrix yarat və göstər
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_,
    )

    disp.plot(xticks_rotation=45)
    plt.title("SVM Emotion Recognition - Confusion Matrix")
    plt.tight_layout()

    # Qrafiki fayl kimi saxla
    results_path = Path("results")
    results_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        results_path / "svm_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    # 11. Öyrədilmiş modeli saxla
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH)
    print("Confusion matrix saved to:", results_path / "svm_confusion_matrix.png")


if __name__ == "__main__":
    main()
