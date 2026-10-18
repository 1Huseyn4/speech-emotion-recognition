"""Predict the emotion of any WAV file with the trained CNN.

Usage:
    python src/predict_emotion.py path/to/audio.wav
    python src/predict_emotion.py path/to/audio.wav --model models/emotion_cnn.pth
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from model import EmotionCNN
from utils import apply_normalizer, load_mfcc_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Predict emotion from a speech WAV file.")
    parser.add_argument("audio", type=Path, help="Path to a .wav file")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/emotion_cnn.pth"),
        help="Path to the trained CNN checkpoint",
    )
    parser.add_argument("--top", type=int, default=3, help="How many top emotions to print")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Train it first with: python src/train_cnn.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    classes = checkpoint["emotion_classes"]
    mean = np.array(checkpoint["mean"], dtype=np.float32)
    std = np.array(checkpoint["std"], dtype=np.float32)

    model = EmotionCNN(
        num_classes=len(classes),
        n_mfcc=checkpoint["n_mfcc"],
        max_len=checkpoint["max_len"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Same preprocessing as training: MFCC -> pad/crop -> normalise with training statistics
    mfcc, length = load_mfcc_matrix(
        args.audio,
        n_mfcc=checkpoint["n_mfcc"],
        max_len=checkpoint["max_len"],
    )
    mfcc = apply_normalizer(mfcc[np.newaxis], np.array([length]), mean, std)
    features = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(features), dim=1)[0]

    top = min(args.top, len(classes))
    values, indices = probabilities.topk(top)

    print(f"Audio file: {args.audio}")
    print(f"Predicted emotion: {classes[indices[0].item()]}")
    print(f"Confidence: {values[0].item():.2%}")
    print("\nTop predictions:")
    for value, index in zip(values, indices):
        print(f"  {classes[index.item()]:<10} {value.item():.2%}")


if __name__ == "__main__":
    main()
