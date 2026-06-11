"""
EmotionDataset
==============
PyTorch Dataset for CREMA-D multimodal emotion recognition.

Each sample returns a dict:
    audio     : Tensor [max_audio_samples]        raw waveform for HuBERT
    frames    : Tensor [num_frames, C, H, W]      normalized face frames
    label     : int                               emotion class index
    has_video : bool                              whether frames are real
    filename  : str
"""

import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Visual Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_visual_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """
    Train  : augmentations + ImageNet normalize
    Val/Test: resize + normalize only
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
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
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Audio Augmentation  (waveform-level, no spectral needed — HuBERT handles it)
# ─────────────────────────────────────────────────────────────────────────────

def augment_audio(wav: np.ndarray, sr: int = 16_000) -> np.ndarray:
    """
    Randomly apply waveform augmentations.
    Called only during training.
    """
    # Gaussian noise
    if random.random() < 0.4:
        noise_amp = random.uniform(0.001, 0.005)
        wav = wav + noise_amp * np.random.randn(*wav.shape).astype(np.float32)

    # Volume scaling
    if random.random() < 0.4:
        wav = wav * random.uniform(0.7, 1.3)

    # Time shift (circular, up to 200ms)
    if random.random() < 0.3:
        shift = random.randint(0, int(0.2 * sr))
        wav = np.roll(wav, shift)

    # Time mask (zero out up to 300ms)
    if random.random() < 0.3:
        max_mask = int(0.3 * sr)
        mask_len = random.randint(0, max_mask)
        start    = random.randint(0, max(0, len(wav) - mask_len))
        wav[start : start + mask_len] = 0.0

    return np.clip(wav, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    """
    PyTorch Dataset for CREMA-D multimodal emotion recognition.

    Args:
        csv_path         : Path to split CSV (train/val/test).
        split            : 'train' | 'val' | 'test'
        config           : Loaded config.yaml as dict.
        visual_transform : Optional transform override.
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        config: Dict,
        visual_transform: Optional[Callable] = None,
    ):
        self.split  = split
        self.config = config
        self.do_audio_aug = (split == "train")

        # ── Load & validate CSV ───────────────────────────────────────
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["label"] >= 0].reset_index(drop=True)

        # ── Config shortcuts ──────────────────────────────────────────
        self.num_frames  = config["visual"]["num_frames"]
        self.image_size  = config["visual"]["image_size"]
        self.sample_rate = config["audio"]["sample_rate"]
        self.max_dur     = config["audio"]["max_duration_sec"]
        self.max_samples = int(self.max_dur * self.sample_rate)
        self.num_classes = config["classifier"]["num_classes"]

        # ── Transforms ────────────────────────────────────────────────
        self.visual_transform = (
            visual_transform or get_visual_transforms(split, self.image_size)
        )

        logger.info(
            f"EmotionDataset [{split:5s}]  {len(self.df):5d} samples  |  "
            f"{self.num_frames} frames  |  {self.max_dur}s audio  |  "
            f"augment={self.do_audio_aug}"
        )

    # ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    # ─────────────────────────────────────────────────────────────────

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load preprocessed WAV → fixed-length float32 tensor [max_samples].
        Falls back to zeros on any read error.
        """
        try:
            wav, _ = sf.read(str(audio_path), dtype="float32")
        except Exception as exc:
            logger.warning(f"Audio read failed [{audio_path}]: {exc}  → zeros")
            return torch.zeros(self.max_samples, dtype=torch.float32)

        # Ensure fixed length
        if len(wav) >= self.max_samples:
            wav = wav[: self.max_samples]
        else:
            wav = np.pad(wav, (0, self.max_samples - len(wav)))

        if self.do_audio_aug:
            wav = augment_audio(wav, self.sample_rate)

        return torch.from_numpy(wav.copy().astype(np.float32))

    # ─────────────────────────────────────────────────────────────────

    def _load_frames(self, frames_dir: str) -> Tuple[torch.Tensor, bool]:
        """
        Load face-crop PNGs → Tensor [num_frames, C, H, W].
        Returns (zero_tensor, False) if directory is missing/empty.
        """
        dummy = torch.zeros(
            self.num_frames, 3, self.image_size, self.image_size,
            dtype=torch.float32,
        )

        if not frames_dir or not Path(frames_dir).exists():
            return dummy, False

        pngs = sorted(Path(frames_dir).glob("*.png"))
        if not pngs:
            return dummy, False

        # Evenly sample num_frames from available PNGs
        indices = np.linspace(0, len(pngs) - 1, self.num_frames, dtype=int)

        tensors: List[torch.Tensor] = []
        for i in indices:
            try:
                img = Image.open(pngs[i]).convert("RGB")
                tensors.append(self.visual_transform(img))
            except Exception as exc:
                logger.warning(f"Frame load failed [{pngs[i]}]: {exc}  → zeros")
                tensors.append(
                    torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
                )

        return torch.stack(tensors, dim=0), True   # [num_frames, C, H, W]

    # ─────────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        audio              = self._load_audio(str(row["audio_path"]))
        frames_dir         = str(row["frames_dir"]) if pd.notna(row["frames_dir"]) else ""
        frames, has_video  = self._load_frames(frames_dir)
        label              = int(row["label"])

        return {
            "audio"    : audio,       # [max_samples]          float32
            "frames"   : frames,      # [num_frames, C, H, W]  float32
            "label"    : label,       # int
            "has_video": has_video,   # bool
            "filename" : str(row["filename"]),
        }

    # ─────────────────────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency class weights for weighted loss.
        Returns Tensor [num_classes].
        """
        counts  = self.df["label"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(np.float32)
        weights = weights / weights.sum() * self.num_classes
        return torch.from_numpy(weights)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader Builder
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate: stack tensors, keep filename as list.
    Standard default_collate would also work but this is explicit.
    """
    return {
        "audio"    : torch.stack([b["audio"]  for b in batch]),    # [B, max_samples]
        "frames"   : torch.stack([b["frames"] for b in batch]),    # [B, T, C, H, W]
        "label"    : torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "has_video": torch.tensor([b["has_video"] for b in batch], dtype=torch.bool),
        "filename" : [b["filename"] for b in batch],
    }


def build_dataloaders(config: Dict, splits_dir: str) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from split CSVs.

    Args:
        config     : Loaded config.yaml dict.
        splits_dir : Directory containing train.csv, val.csv, test.csv.

    Returns:
        Dict with keys 'train', 'val', 'test'.
    """
    splits_path  = Path(splits_dir)
    batch_size   = config["training"]["batch_size"]
    num_workers  = config["training"]["num_workers"]

    loaders: Dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        csv_path = splits_path / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        dataset = EmotionDataset(
            csv_path=str(csv_path),
            split=split,
            config=config,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            collate_fn  = collate_fn,
            pin_memory  = True,
            drop_last   = (split == "train"),
            persistent_workers = (num_workers > 0),
        )

        logger.info(
            f"DataLoader [{split:5s}]  {len(dataset):5d} samples  "
            f"→  {len(loaders[split])} batches  (bs={batch_size})"
        )

    return loaders

# ─────────────────────────────────────────────────────────────────────────────
# Fast Dataset  (loads pre-cached features instead of raw audio/video)
# ─────────────────────────────────────────────────────────────────────────────

class CachedEmotionDataset(Dataset):
    """
    Loads pre-extracted HuBERT audio features  [1536]
    and EfficientNet frame features            [num_frames, 1536]
    from disk. ~50x faster than raw EmotionDataset.
    """

    def __init__(self, csv_path: str, split: str, config: Dict):
        self.split      = split
        self.df         = pd.read_csv(csv_path)
        self.df         = self.df[self.df["label"] >= 0].reset_index(drop=True)
        self.num_frames = config["visual"]["num_frames"]
        self.do_aug     = (split == "train")

        processed_dir    = config["data"]["processed_dir"]
        feat_dir         = Path(processed_dir) / "features"
        self.audio_feat  = feat_dir / "audio"
        self.frame_feat  = feat_dir / "frames"

        if not self.audio_feat.exists():
            raise FileNotFoundError(
                f"Feature cache not found: {self.audio_feat}\n"
                "Run: python scripts/00_cache_features.py"
            )

        self.num_classes = config["classifier"]["num_classes"]
        logger.info(
            f"CachedEmotionDataset [{split:5s}]  {len(self.df):5d} samples  "
            f"(cached features)"
        )

    def __len__(self):
        return len(self.df)

    def get_class_weights(self) -> torch.Tensor:
        counts  = self.df["label"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(np.float32)
        weights = weights / weights.sum() * self.num_classes
        return torch.from_numpy(weights)

    def __getitem__(self, idx: int) -> Dict:
        row   = self.df.iloc[idx]
        fname = str(row["filename"])

        # ── Audio feature ─────────────────────────────────────────
        a_path = self.audio_feat / f"{fname}.pt"
        audio_feat = torch.load(a_path, map_location="cpu", weights_only=True) \
            if a_path.exists() else torch.zeros(1536)

        # ── Frame features ────────────────────────────────────────
        f_path = self.frame_feat / f"{fname}.pt"
        if f_path.exists():
            frame_feat = torch.load(f_path, map_location="cpu", weights_only=True)  # [T, 1536]
            if frame_feat.shape[0] != self.num_frames:
                idx_f = np.linspace(0, frame_feat.shape[0]-1, self.num_frames, dtype=int)
                frame_feat = frame_feat[idx_f]
        else:
            frame_feat = torch.zeros(self.num_frames, 1536)

        # Light augmentation on features (training only)
        if self.do_aug:
            if torch.rand(1).item() < 0.3:
                audio_feat = audio_feat + 0.01 * torch.randn_like(audio_feat)
            if torch.rand(1).item() < 0.3:
                frame_feat = frame_feat + 0.01 * torch.randn_like(frame_feat)

        return {
            "audio_feat" : audio_feat.float(),      # [1536]
            "frame_feat" : frame_feat.float(),      # [num_frames, 1536]
            "label"      : int(row["label"]),
            "has_video"  : frame_feat.abs().sum().item() > 0,
            "filename"   : fname,
        }


def cached_collate_fn(batch: List[Dict]) -> Dict:
    return {
        "audio_feat": torch.stack([b["audio_feat"] for b in batch]),
        "frame_feat": torch.stack([b["frame_feat"] for b in batch]),
        "label"     : torch.tensor([b["label"]     for b in batch], dtype=torch.long),
        "has_video" : torch.tensor([b["has_video"] for b in batch], dtype=torch.bool),
        "filename"  : [b["filename"] for b in batch],
    }


def build_cached_dataloaders(config: Dict, splits_dir: str) -> Dict[str, DataLoader]:
    splits_path = Path(splits_dir)
    batch_size  = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        csv_path = splits_path / f"{split}.csv"
        ds       = CachedEmotionDataset(str(csv_path), split, config)
        loaders[split] = DataLoader(
            ds,
            batch_size         = batch_size,
            shuffle            = (split == "train"),
            num_workers        = num_workers,
            collate_fn         = cached_collate_fn,
            pin_memory         = True,
            drop_last          = (split == "train"),
            persistent_workers = (num_workers > 0),
        )
        logger.info(
            f"CachedDataLoader [{split:5s}]  {len(ds):5d} samples  "
            f"→  {len(loaders[split])} batches"
        )
    return loaders