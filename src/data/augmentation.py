"""
augmentation.py
===============
Standalone augmentation utilities (for import from other modules).
"""

import random

import numpy as np
import torch
from torchvision import transforms


class AudioAugmentor:
    """
    Waveform-level audio augmentation pipeline.

    Usage:
        aug = AudioAugmentor(p=0.5)
        wav_aug = aug(wav_tensor)   # wav_tensor: float32 Tensor [N]
    """

    def __init__(self, p: float = 0.5, sample_rate: int = 16_000):
        self.p           = p
        self.sample_rate = sample_rate

    def add_gaussian_noise(self, wav: torch.Tensor, snr_db: float = 30.0) -> torch.Tensor:
        signal_power = wav.pow(2).mean().clamp(min=1e-9)
        snr_linear   = 10 ** (snr_db / 10.0)
        noise_power  = signal_power / snr_linear
        noise        = torch.randn_like(wav) * noise_power.sqrt()
        return (wav + noise).clamp(-1.0, 1.0)

    def scale_volume(self, wav: torch.Tensor, low: float = 0.7, high: float = 1.3) -> torch.Tensor:
        return (wav * random.uniform(low, high)).clamp(-1.0, 1.0)

    def time_shift(self, wav: torch.Tensor, max_ms: float = 200.0) -> torch.Tensor:
        max_shift = int(max_ms * self.sample_rate / 1000)
        return torch.roll(wav, random.randint(0, max_shift))

    def time_mask(self, wav: torch.Tensor, max_ms: float = 300.0) -> torch.Tensor:
        max_len = int(max_ms * self.sample_rate / 1000)
        length  = random.randint(0, max_len)
        start   = random.randint(0, max(0, len(wav) - length))
        wav     = wav.clone()
        wav[start : start + length] = 0.0
        return wav

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return wav
        if random.random() < 0.4:
            wav = self.add_gaussian_noise(wav)
        if random.random() < 0.4:
            wav = self.scale_volume(wav)
        if random.random() < 0.3:
            wav = self.time_shift(wav)
        if random.random() < 0.3:
            wav = self.time_mask(wav)
        return wav


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])