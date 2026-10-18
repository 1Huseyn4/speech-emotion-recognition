
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4800, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def extract_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
    )

    target_length = 200

    if mfcc.shape[1] < target_length:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, target_length - mfcc.shape[1])),
        )
    else:
        mfcc = mfcc[:, :target_length]

    return torch.tensor(
        mfcc,
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)


def main():
    model_path = Path("models/emotion_cnn.pth")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Datasetdəki WAV fayllarından birini avtomatik seç
    audio_files = list(Path("data").rglob("*.wav"))

    if not audio_files:
        raise FileNotFoundError("No WAV files found in data/")

    audio_path = audio_files[0]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    emotion_classes = checkpoint["emotion_classes"]

    model = EmotionCNN(num_classes=len(emotion_classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features = extract_mfcc(audio_path).to(device)

    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    emotion = emotion_classes[prediction.item()]

    print(f"Audio file: {audio_path}")
    print(f"Predicted emotion: {emotion}")
    print(f"Confidence: {confidence.item():.2%}")


if __name__ == "__main__":
    main()
