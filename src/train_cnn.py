from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt



# Faylların yerləşdiyi qovluq
DATA_DIR = Path("data/processed/cnn_features")

# Etiketləri oxu
labels_df = pd.read_csv(DATA_DIR / "labels.csv")

# GPU varsa GPU, yoxdursa CPU seç
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Total samples:", len(labels_df))
print("Emotions:", sorted(labels_df["emotion"].unique()))

# Emosiya adlarını rəqəmlərə çevir
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels_df["emotion"])

# Bütün MFCC fayllarını oxu
X = np.array([
    np.load(DATA_DIR / f"{index}.npy")
    for index in labels_df["index"]
])

# CNN üçün kanal ölçüsü əlavə et
X = X[:, np.newaxis, :, :]

# Təlim və test hissələrinə ayır
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X shape:", X.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Number of classes:", len(label_encoder.classes_)) 

# CNN arxitekturasını əlavə et


class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 3 * 50, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = EmotionCNN(num_classes=len(label_encoder.classes_)).to(device)

print(model)


# NumPy massivlərini PyTorch tensorlarına çevir
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Məlumatları GPU-ya göndər
X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)

y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

print("Training tensor:", X_train_tensor.shape)
print("Test tensor:", X_test_tensor.shape)
print("Tensors are on:", X_train_tensor.device)


# Səhvin ölçülməsi üçün funksiya
criterion = nn.CrossEntropyLoss()

# Modelin parametrlərini yeniləmək üçün optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Loss function:", criterion)
print("Optimizer:", optimizer)



from torch.utils.data import TensorDataset, DataLoader

# Məlumatları kiçik qruplara böl
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# Modeli öyrət
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        # Əvvəlki gradientləri sıfırla
        optimizer.zero_grad()

        # Modelin proqnozunu al
        outputs = model(X_batch)

        # Səhvi hesabla
        loss = criterion(outputs, y_batch)

        # Səhvin model parametrlərinə təsirini hesabla
        loss.backward()

        # Model parametrlərini yenilə
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
    )


# Modeli qiymətləndirmə rejiminə keçir
model.eval()

# Gradient hesablamasını söndür
with torch.no_grad():
    # Test məlumatları üzrə proqnoz al
    outputs = model(X_test_tensor)

    # Hər nümunə üçün ən yüksək ehtimallı sinfi seç
    predictions = torch.argmax(outputs, dim=1)

# Nəticələri CPU-da NumPy massivlərinə çevir
y_true = y_test_tensor.cpu().numpy()
y_pred = predictions.cpu().numpy()

# Ümumi dəqiqliyi hesabla
accuracy = accuracy_score(y_true, y_pred)

print(f"\nTest Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=np.arange(len(label_encoder.classes_)),
    display_labels=label_encoder.classes_,
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("CNN Emotion Recognition - Confusion Matrix")
plt.tight_layout()
plt.savefig("data/processed/cnn_confusion_matrix.png")
plt.show()


from pathlib import Path

# Model üçün qovluq yarat
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Öyrədilmiş modeli və emosiya adlarını saxla
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "emotion_classes": label_encoder.classes_.tolist(),
    },
    MODEL_DIR / "emotion_cnn.pth"
)

print("Model saved to models/emotion_cnn.pth")
