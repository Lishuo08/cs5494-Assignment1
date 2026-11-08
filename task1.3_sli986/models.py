"""Convolutional MNIST GAN architectures from section 4 of 实验方案.md."""

import torch
from torch import nn


class Generator(nn.Module):
    """Map [B, 100] Gaussian noise to [B, 1, 28, 28] images in [-1, 1]."""

    def __init__(self, latent_dim: int = 100, use_batchnorm: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_batchnorm = use_batchnorm

        self.project = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.output(self.upsample(self.project(z)))


class Discriminator(nn.Module):
    """Map [B, 1, 28, 28] images to [B, 1] logits for BCEWithLogitsLoss."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
