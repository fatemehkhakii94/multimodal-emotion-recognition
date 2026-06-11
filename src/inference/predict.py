"""
EmotionPredictor
================
End-to-end inference engine.
Supports:
  - predict_from_processed() : pre-cropped frames + 16kHz WAV  (fastest)
  - predict_from_file()      : raw video file (runs full pipeline)
  - predict_from_arrays()    : numpy audio + BGR frame list
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import librosa
import numpy as np
import soundfile as sf
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import HubertModel

from src.data.preprocess import FaceDetector
from src.models.multimodal_model import CachedMultimodalEmotionModel

logger = logging.getLogger(__name__)

EMOTION_NAMES  = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
EMOTION_LABELS = {
    "ANG": "Angry",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}
EMOTION_COLORS = {
    "ANG": "#E74C3C",
    "DIS": "#E67E22",
    "FEA": "#F1C40F",
    "HAP": "#2ECC71",
    "NEU": "#3498DB",
    "SAD": "#9B59B6",
}

EXPERIMENT_MAP = {
    "cross_modal": {"fusion": "cross_modal", "mode": "multimodal"},
    "early"      : {"fusion": "early",       "mode": "multimodal"},
    "late"       : {"fusion": "late",        "mode": "multimodal"},
    "audio_only" : {"fusion": "cross_modal", "mode": "audio_only"},
    "visual_only": {"fusion": "cross_modal", "mode": "visual_only"},
}


class EmotionPredictor:
    """
    Args:
        config   : loaded config.yaml dict
        run_name : experiment name  (cross_modal | early | late | audio_only | visual_only)
        device   : 'cuda' | 'cpu' | None (auto)
    """

    def __init__(
        self,
        config:   Dict,
        run_name: str           = "cross_modal",
        device:   Optional[str] = None,
    ):
        self.config   = config
        self.run_name = run_name
        self.device   = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        exp        = EXPERIMENT_MAP.get(run_name, EXPERIMENT_MAP["cross_modal"])
        self.mode  = exp["mode"]
        self.fusion = exp["fusion"]

        self.num_frames  = config["visual"]["num_frames"]
        self.image_size  = config["visual"]["image_size"]
        self.sample_rate = config["audio"]["sample_rate"]
        self.max_dur     = config["audio"]["max_duration_sec"]
        self.max_samples = int(self.max_dur * self.sample_rate)

        self._load_backbones()
        self._load_classifier()

        self.visual_tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.face_detector = FaceDetector()

        logger.info(
            f"EmotionPredictor ready  "
            f"run={run_name}  mode={self.mode}  device={self.device}"
        )

    # ─────────────────────────────────────────────────────────────

    def _load_backbones(self):
        logger.info("Loading HuBERT …")
        self.hubert = HubertModel.from_pretrained(
            self.config["audio"]["model_name"]
        ).to(self.device)
        self.hubert.eval()
        for p in self.hubert.parameters():
            p.requires_grad = False

        logger.info("Loading EfficientNet-B3 …")
        self.effnet = timm.create_model(
            self.config["visual"]["efficientnet_variant"],
            pretrained  = True,
            num_classes = 0,
            global_pool = "avg",
        ).to(self.device)
        self.effnet.eval()
        for p in self.effnet.parameters():
            p.requires_grad = False

    def _load_classifier(self):
        ckpt_path = (
            Path(self.config["paths"]["checkpoints"]) / f"{self.run_name}_best.pt"
        )
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Run: python scripts/02_train.py first."
            )
        self.classifier = CachedMultimodalEmotionModel(
            self.config, fusion_type=self.fusion
        ).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.classifier.load_state_dict(state["model_state"])
        self.classifier.eval()
        logger.info(f"Loaded [{self.run_name}]  epoch={state['epoch']}")

    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _audio_to_feat(self, wav: np.ndarray) -> torch.Tensor:
        """float32 waveform → Tensor [1536]"""
        if len(wav) > self.max_samples:
            wav = wav[: self.max_samples]
        else:
            wav = np.pad(wav, (0, self.max_samples - len(wav)))
        wav_t  = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(self.device)
        hidden = self.hubert(input_values=wav_t).last_hidden_state     # [1, T, 768]
        mean   = hidden.mean(dim=1)
        std    = hidden.std(dim=1).clamp(min=1e-6)
        return torch.cat([mean, std], dim=-1).squeeze(0).cpu()          # [1536]

    @torch.no_grad()
    def _rgb_images_to_feat(self, images_rgb: List[np.ndarray]) -> torch.Tensor:
        """List of RGB uint8 images (already face-cropped or full frames) → [T, 1536]"""
        if not images_rgb:
            return torch.zeros(self.num_frames, 1536)
        indices = np.linspace(0, len(images_rgb) - 1, self.num_frames, dtype=int)
        tensors = []
        for i in indices:
            img = Image.fromarray(images_rgb[int(i)])
            tensors.append(self.visual_tf(img))
        batch = torch.stack(tensors).to(self.device)     # [T, 3, H, W]
        return self.effnet(batch).cpu()                   # [T, 1536]

    @torch.no_grad()
    def _classify(
        self,
        audio_feat: torch.Tensor,   # [1536]
        frame_feat: torch.Tensor,   # [T, 1536]
    ) -> Dict:
        a      = audio_feat.unsqueeze(0).to(self.device)
        f      = frame_feat.unsqueeze(0).to(self.device)
        logits, _ = self.classifier(a, f, mode=self.mode)
        probs  = F.softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()
        idx    = int(probs.argmax())
        return {
            "emotion"      : EMOTION_NAMES[idx],
            "emotion_full" : EMOTION_LABELS[EMOTION_NAMES[idx]],
            "confidence"   : float(probs[idx]),
            "probs"        : {EMOTION_NAMES[i]: float(probs[i])
                              for i in range(len(EMOTION_NAMES))},
            "label"        : idx,
        }

    # ─────────────────────────────────────────────────────────────
    # Public prediction API
    # ─────────────────────────────────────────────────────────────

    def predict_from_processed(
        self,
        audio_path: str,
        frames_dir: str,
    ) -> Dict:
        """
        Fastest path: uses pre-processed 16kHz WAV + face-cropped PNGs.
        Used in dataset demo with CREMA-D processed data.
        """
        # Audio
        try:
            wav, _ = sf.read(str(audio_path), dtype="float32")
        except Exception as e:
            logger.warning(f"Audio read failed: {e}")
            wav = np.zeros(self.max_samples, dtype=np.float32)

        audio_feat = self._audio_to_feat(wav)

        # Frames — already face-cropped PNGs
        images_rgb = []
        frames_p   = Path(frames_dir)
        if frames_p.exists():
            for png in sorted(frames_p.glob("*.png")):
                img = cv2.imread(str(png))
                if img is not None:
                    images_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        frame_feat = self._rgb_images_to_feat(images_rgb)

        result          = self._classify(audio_feat, frame_feat)
        result["wav"]   = wav
        result["faces"] = images_rgb   # for visualization
        return result

    def predict_from_file(self, video_path: str) -> Dict:
        """
        Full pipeline on a raw video file (FLV, MP4, AVI, …).
        Runs face detection internally.
        """
        video_path = str(video_path)

        # Extract raw frames
        cap    = cv2.VideoCapture(video_path)
        frames_bgr = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_bgr.append(frame)
        cap.release()

        # Extract audio
        try:
            wav, _ = librosa.load(video_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            logger.warning(f"Audio load failed: {e}")
            wav = np.zeros(self.max_samples, dtype=np.float32)

        audio_feat = self._audio_to_feat(wav)

        # Face detection → RGB crops
        images_rgb = []
        if frames_bgr:
            indices = np.linspace(0, len(frames_bgr) - 1, self.num_frames, dtype=int)
            for i in indices:
                face = self.face_detector.get_face(frames_bgr[int(i)], size=self.image_size)
                images_rgb.append(face)

        frame_feat = self._rgb_images_to_feat(images_rgb)

        result          = self._classify(audio_feat, frame_feat)
        result["wav"]   = wav
        result["faces"] = images_rgb
        return result

    def close(self):
        self.face_detector.close()