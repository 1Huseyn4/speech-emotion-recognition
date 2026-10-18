"""CNN architecture shared by training and prediction (single source of truth)."""

import torch
import torch.nn as nn

from utils import MAX_LEN, N_MFCC


class EmotionCNN(nn.Module):
    """Small 2D CNN that treats the MFCC matrix (n_mfcc x frames) as an image."""

    def __init__(self, num_classes, n_mfcc=N_MFCC, max_len=MAX_LEN, dropout=0.4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 8)),

            nn.Flatten(),
        )

        # Work out the flattened size automatically instead of hard-coding it
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, 1, n_mfcc, max_len)).shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
