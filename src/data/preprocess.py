"""
CREMA-D Preprocessing Pipeline
================================
Visual : FLV → 15 evenly-spaced frames → OpenCV face crop → 224×224 PNG
Audio  : WAV → mono 16 kHz, padded/trimmed to max_duration   → WAV
Splits : Actor-based 70 / 15 / 15  (zero actor leakage)
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Face Detector  (OpenCV Haar Cascade — no external deps, works everywhere)
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """
    OpenCV Haar-Cascade face detector with automatic centre-crop fallback.
    Always returns an (H, W, 3) uint8 RGB image.
    """

    def __init__(self, min_detection_confidence: float = 0.4):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from: {cascade_path}")
        logger.info(f"FaceDetector ready  (Haar cascade  ←  {cascade_path})")

    def get_face(
        self,
        bgr: np.ndarray,
        pad_ratio: float = 0.25,
        size: int = 224,
    ) -> np.ndarray:
        """
        Detect largest frontal face, add padding, resize to (size, size).
        Falls back to square centre-crop if no face is detected.

        Args:
            bgr:       BGR image (OpenCV format).
            pad_ratio: Fractional padding added around the bounding box.
            size:      Output square side length in pixels.

        Returns:
            RGB image of shape (size, size, 3).
        """
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # improves detection in dark/bright frames

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) > 0:
            # Pick the largest detected face
            areas = [fw * fh for (_, _, fw, fh) in faces]
            fx, fy, fw, fh = faces[int(np.argmax(areas))]

            # Add padding
            px = int(fw * pad_ratio)
            py = int(fh * pad_ratio)
            x1 = max(0, fx - px)
            y1 = max(0, fy - py)
            x2 = min(w, fx + fw + 2 * px)
            y2 = min(h, fy + fh + 2 * py)

            crop = bgr[y1:y2, x1:x2]
            if crop.size > 0:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                return cv2.resize(rgb, (size, size))

        # Fallback: square centre-crop of full frame
        side = min(h, w)
        cy, cx = h // 2, w // 2
        half   = side // 2
        crop   = bgr[max(0, cy - half) : cy + half, max(0, cx - half) : cx + half]
        rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (size, size))

    def close(self):
        pass   # nothing to release for Haar cascade


# ─────────────────────────────────────────────────────────────────────────────
# Frame Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: Path,
    num_frames: int                       = 15,
    face_detector: Optional[FaceDetector] = None,
    image_size: int                       = 224,
) -> int:
    """
    Read all video frames sequentially, sample ``num_frames`` evenly-spaced
    ones, crop face, and save as PNG.

    Returns:
        Number of frames saved  (0 on failure).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return 0

    raw: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw.append(frame)
    cap.release()

    if not raw:
        logger.warning(f"No frames decoded: {video_path}")
        return 0

    total   = len(raw)
    indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)

    saved = 0
    for out_idx, src_idx in enumerate(indices):
        bgr  = raw[int(src_idx)]
        face = (
            face_detector.get_face(bgr, size=image_size)
            if face_detector
            else cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (image_size, image_size))
        )
        # face is RGB → convert to BGR for cv2.imwrite
        out_path = output_dir / f"{out_idx:04d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        saved += 1

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Audio Processing
# ─────────────────────────────────────────────────────────────────────────────

def process_audio(
    audio_path: str,
    output_path: Path,
    target_sr: int      = 16_000,
    max_duration: float = 6.0,
) -> bool:
    """
    Load, resample (mono 16 kHz), pad/trim, write 16-bit PCM WAV.

    Returns:
        True on success, False on failure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        wav, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
    except Exception as exc:
        logger.warning(f"Audio load failed [{audio_path}]: {exc}")
        return False

    max_samples = int(max_duration * target_sr)
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    else:
        wav = np.pad(wav, (0, max_samples - len(wav)))

    try:
        sf.write(str(output_path), wav, target_sr, subtype="PCM_16")
    except Exception as exc:
        logger.warning(f"Audio write failed [{output_path}]: {exc}")
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Actor-Based Splits
# ─────────────────────────────────────────────────────────────────────────────

def create_splits(
    samples: List[Dict],
    splits_dir: str,
    emotion_map: Dict[str, int],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Partition by actor ID so no actor appears in more than one split.
    Saves train.csv / val.csv / test.csv to ``splits_dir``.
    """
    out = Path(splits_dir)
    out.mkdir(parents=True, exist_ok=True)

    actors = sorted({s["actor_id"] for s in samples})
    random.seed(seed)
    random.shuffle(actors)

    n_train = int(len(actors) * train_ratio)
    n_val   = int(len(actors) * val_ratio)

    train_set = set(actors[:n_train])
    val_set   = set(actors[n_train : n_train + n_val])
    test_set  = set(actors[n_train + n_val :])

    logger.info(
        f"Actor split  —  train: {len(train_set)}  "
        f"val: {len(val_set)}  test: {len(test_set)}"
    )

    buckets: Dict[str, list] = {"train": [], "val": [], "test": []}

    for s in samples:
        split = (
            "train" if s["actor_id"] in train_set
            else "val" if s["actor_id"] in val_set
            else "test"
        )
        buckets[split].append({
            "filename"  : s["filename"],
            "audio_path": s.get("processed_audio_path", s["audio_path"]),
            "frames_dir": s.get("processed_frames_dir", ""),
            "emotion"   : s["emotion"],
            "label"     : emotion_map.get(s["emotion"], -1),
            "actor_id"  : s["actor_id"],
            "has_video" : s.get("has_video", False),
            "n_frames"  : s.get("n_frames", 0),
        })

    dfs: Dict[str, pd.DataFrame] = {}
    for name, rows in buckets.items():
        df = pd.DataFrame(rows)
        df.to_csv(out / f"{name}.csv", index=False)
        dfs[name] = df
        logger.info(f"  {name:5s}: {len(df)} samples  →  {out / (name + '.csv')}")

    return dfs


# ─────────────────────────────────────────────────────────────────────────────
# Main Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dataset(
    samples: List[Dict],
    processed_dir: str,
    config: Dict,
    resume: bool = True,
) -> List[Dict]:
    """
    For every sample:
      1. Process audio  → ``processed_dir/audio/{filename}.wav``
      2. Extract frames → ``processed_dir/frames/{filename}/*.png``

    Returns:
        Updated sample list with keys:
        ``processed_audio_path``, ``processed_frames_dir``, ``n_frames``.
    """
    proc    = Path(processed_dir).resolve()
    aud_dir = proc / "audio"
    frm_dir = proc / "frames"
    aud_dir.mkdir(parents=True, exist_ok=True)
    frm_dir.mkdir(parents=True, exist_ok=True)

    num_frames = config["visual"]["num_frames"]
    img_size   = config["visual"]["image_size"]
    target_sr  = config["audio"]["sample_rate"]
    max_dur    = config["audio"]["max_duration_sec"]

    detector = FaceDetector(min_detection_confidence=0.4)

    processed: List[Dict] = []
    n_skipped = 0

    logger.info(f"Processing {len(samples)} samples  (resume={resume}) …")

    for sample in tqdm(samples, desc="Preprocessing", unit="clip", ncols=80):
        fname = sample["filename"]

        # ── Audio ────────────────────────────────────────────────────
        audio_out = aud_dir / f"{fname}.wav"
        if resume and audio_out.exists() and audio_out.stat().st_size > 1_000:
            audio_ok = True
        else:
            audio_ok = process_audio(sample["audio_path"], audio_out, target_sr, max_dur)

        if not audio_ok:
            n_skipped += 1
            continue

        # ── Frames ───────────────────────────────────────────────────
        frame_dir     = frm_dir / fname
        existing_pngs = list(frame_dir.glob("*.png")) if frame_dir.exists() else []

        if resume and len(existing_pngs) > 0:
            n_saved = len(existing_pngs)
        elif sample["has_video"] and sample["video_path"]:
            n_saved = extract_frames(
                sample["video_path"], frame_dir, num_frames, detector, img_size
            )
        else:
            n_saved = 0

        processed.append({
            **sample,
            "processed_audio_path": str(audio_out),
            "processed_frames_dir": str(frame_dir) if n_saved > 0 else "",
            "n_frames"            : n_saved,
        })

    detector.close()

    n_with_video = sum(1 for s in processed if s["n_frames"] > 0)
    logger.info(
        f"Done  —  {len(processed)} processed  "
        f"({n_with_video} with frames, {n_skipped} skipped)"
    )
    return processed
# ─────────────────────────────────────────────────────────────────────────────
# Feature Caching  (HuBERT + EfficientNet)
# ─────────────────────────────────────────────────────────────────────────────

import torch

def extract_and_cache_features(
    splits_dir:    str,
    processed_dir: str,
    config:        Dict,
    device:        str = "cuda",
) -> None:
    """
    Pre-extract HuBERT audio features and EfficientNet frame features,
    save as .pt tensors to disk.  Run once; skip if already cached.

    Outputs:
        processed_dir/features/audio/{filename}.pt   → Tensor [1536]
        processed_dir/features/frames/{filename}.pt  → Tensor [T, 1536]
    """
    import timm
    from transformers import HubertModel
    from src.data.dataset import EmotionDataset, collate_fn
    from torch.utils.data import DataLoader

    feat_dir   = Path(processed_dir) / "features"
    audio_feat = feat_dir / "audio"
    frame_feat = feat_dir / "frames"
    audio_feat.mkdir(parents=True, exist_ok=True)
    frame_feat.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Feature extraction device: {dev}")

    # ── Load backbones ────────────────────────────────────────────
    logger.info("Loading HuBERT …")
    hubert = HubertModel.from_pretrained(config["audio"]["model_name"]).to(dev)
    hubert.eval()

    logger.info("Loading EfficientNet-B3 …")
    effnet = timm.create_model(
        config["visual"]["efficientnet_variant"],
        pretrained  = True,
        num_classes = 0,
        global_pool = "avg",
    ).to(dev)
    effnet.eval()

    # ── Dataset (all splits combined, no augmentation) ────────────
    all_rows = []
    for split in ("train", "val", "test"):
        csv_path = Path(splits_dir) / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_rows.append(df)
    all_df = pd.concat(all_rows, ignore_index=True)
    logger.info(f"Total clips to cache: {len(all_df)}")

    target_sr  = config["audio"]["sample_rate"]
    max_dur    = config["audio"]["max_duration_sec"]
    max_samp   = int(target_sr * max_dur)
    num_frames = config["visual"]["num_frames"]
    img_size   = config["visual"]["image_size"]

    from torchvision import transforms
    from PIL import Image

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    n_cached = 0
    n_skip   = 0

    with torch.no_grad():
        for _, row in tqdm(all_df.iterrows(), total=len(all_df),
                           desc="Caching features", ncols=80):
            fname = str(row["filename"])

            a_out = audio_feat / f"{fname}.pt"
            f_out = frame_feat / f"{fname}.pt"
            if a_out.exists() and f_out.exists():
                n_skip += 1
                continue

            # ── Audio feature ─────────────────────────────────────
            if not a_out.exists():
                try:
                    wav, _ = sf.read(str(row["audio_path"]), dtype="float32")
                    if len(wav) > max_samp:
                        wav = wav[:max_samp]
                    else:
                        wav = np.pad(wav, (0, max_samp - len(wav)))
                    wav_t  = torch.from_numpy(wav).unsqueeze(0).to(dev)   # [1, N]
                    hidden = hubert(input_values=wav_t).last_hidden_state  # [1, T, 768]
                    mean   = hidden.mean(dim=1)                             # [1, 768]
                    std    = hidden.std(dim=1).clamp(min=1e-6)
                    pooled = torch.cat([mean, std], dim=-1).squeeze(0).cpu()  # [1536]
                    torch.save(pooled, a_out)
                except Exception as e:
                    logger.warning(f"Audio feat failed [{fname}]: {e}")
                    torch.save(torch.zeros(1536), a_out)

            # ── Frame features ────────────────────────────────────
            if not f_out.exists():
                frames_dir = Path(str(row["frames_dir"])) if pd.notna(row["frames_dir"]) else Path("")
                pngs = sorted(frames_dir.glob("*.png")) if frames_dir.exists() else []

                if pngs:
                    indices = np.linspace(0, len(pngs)-1, num_frames, dtype=int)
                    imgs = []
                    for i in indices:
                        try:
                            img = Image.open(pngs[i]).convert("RGB")
                            imgs.append(tf(img))
                        except:
                            imgs.append(torch.zeros(3, img_size, img_size))
                    batch_frames = torch.stack(imgs).to(dev)              # [T, 3, H, W]
                    feats        = effnet(batch_frames)                    # [T, 1536]
                    torch.save(feats.cpu(), f_out)
                else:
                    torch.save(torch.zeros(num_frames, 1536), f_out)

            n_cached += 1

    logger.info(f"Done  —  {n_cached} cached  {n_skip} skipped")
    del hubert, effnet
    torch.cuda.empty_cache()