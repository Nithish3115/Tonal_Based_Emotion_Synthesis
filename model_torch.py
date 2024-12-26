"""
Author: Jeffrey Luo, Monta Vista High School, Cupertino, CA
Date Created: 09/2023
Copyright (c) 2023 Jeff Luo
License: MIT
"""
import torch
from torch import nn

class Audio2EmotionModel(nn.Module):
    """
    Audio to emotion model based on VGG. This model takes vectors of spectrograms as
    input and output the emotion scores in 8 dimensional statistics (valence,energy,tension,anger,fear,
    happy,sad,tender).

    Inputs:
        Batch of spectrograms vectors, default shape (B, 1, 256, 1292)

    Outputs:
        Batch of emotion scores in (B, 8)
    """
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            # layer 1
            nn.Conv2d(1, 64, kernel_size=(5,5), stride=2, padding='valid'),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # layer 2
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # layer 3
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.3),

            # layer 4
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # layer 5
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            #layer 6
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.3),

            # layer 7
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # layer 8
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # layer 9
            nn.Conv2d(256, 384, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(384),
            nn.ReLU(),

            # layer 10
            nn.Conv2d(384, 512, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # layer 11
            nn.Conv2d(512, 256, kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # layer 12
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, 8)

    def forward(self, x):
        x = self.layers(x)      # [B, 256, 1, 1]
        # reshape for linear head
        x = torch.squeeze(x)    # [B, 256]
        x = self.head(x)        # [B, 8]
        return x
