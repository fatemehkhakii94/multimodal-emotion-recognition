"""
CREMA-D Dataset Downloader
==========================
Crowd-sourced Emotional Multimodal Actors Dataset
  - 7,442 clips  |  91 actors  |  6 emotions
  - Source: https://github.com/CheyneyComputerScience/CREMA-D
"""

import sys
import shutil
import subprocess
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

CREMAD_REPO_URL     = "https://github.com/CheyneyComputerScience/CREMA-D.git"
VALID_EMOTIONS      = {"ANG", "DIS", "FEA", "HAP", "NEU", "SAD"}
EXPECTED_CLIP_COUNT = 7_442


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _run(cmd: List[str]) -> None:
    """Log and execute a shell command, raise on failure."""
    logger.info("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ─── Public API ───────────────────────────────────────────────────────────────

def download_cremad(cremad_dir: str = "data/raw/CREMAD", force: bool = False) -> str:
    """
    Download CREMA-D from GitHub using a sparse depth-1 clone.
    Only ``AudioWAV/`` and ``VideoFlash/`` are fetched (~4-5 GB).

    Args:
        cremad_dir: Destination directory (created if absent).
        force:      Delete and re-download even if files already exist.

    Returns:
        Resolved path to the CREMA-D directory.
    """
    root      = Path(cremad_dir).resolve()
    audio_dir = root / "AudioWAV"
    video_dir = root / "VideoFlash"

    # ── Already downloaded? ───────────────────────────────────────────
    if not force and root.exists():
        n_wav = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0
        n_flv = len(list(video_dir.glob("*.flv"))) if video_dir.exists() else 0
        if n_wav >= EXPECTED_CLIP_COUNT:
            logger.info(f"CREMA-D already present  ({n_wav} audio  {n_flv} video)  →  {root}")
            return str(root)
        elif n_wav > 0:
            logger.warning(
                f"Partial download found ({n_wav} WAV files). "
                "Continuing with available files. Use --force to re-download."
            )
            return str(root)

    # ── Force-clean ───────────────────────────────────────────────────
    if force and root.exists():
        logger.info(f"--force : removing {root}")
        shutil.rmtree(root)

    # ── Git check ─────────────────────────────────────────────────────
    if not _git_available():
        logger.error("git not found. Install git and retry, or manually download:")
        logger.error("  https://github.com/CheyneyComputerScience/CREMA-D")
        logger.error(f"  Place AudioWAV/ and VideoFlash/ inside: {root}")
        sys.exit(1)

    root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Downloading CREMA-D  (sparse clone, depth=1)")
    logger.info(f"  Target : {root}")
    logger.info("  Size   : ~4-5 GB  (AudioWAV + VideoFlash only)")
    logger.info("  ETA    : 15-45 min depending on bandwidth")
    logger.info("=" * 60)

    try:
        # Step 1: shallow clone, skip all blobs initially
        _run([
            "git", "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--depth", "1",
            CREMAD_REPO_URL,
            str(root),
        ])

        # Step 2: configure cone-mode sparse checkout
        _run(["git", "-C", str(root), "sparse-checkout", "init", "--cone"])
        _run(["git", "-C", str(root), "sparse-checkout", "set", "AudioWAV", "VideoFlash"])

        # Step 3: checkout (fetches only selected directories' blobs)
        logger.info("  Downloading AudioWAV + VideoFlash blobs …  (this is the slow part)")
        _run(["git", "-C", str(root), "checkout"])

        # ── Report ────────────────────────────────────────────────────
        n_wav = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0
        n_flv = len(list(video_dir.glob("*.flv"))) if video_dir.exists() else 0
        logger.info(f"Download done  →  {n_wav} audio  {n_flv} video")

        if n_wav < EXPECTED_CLIP_COUNT:
            logger.warning(
                f"Expected {EXPECTED_CLIP_COUNT} WAV files, got {n_wav}. "
                "Some files may be missing."
            )

    except subprocess.CalledProcessError as exc:
        logger.error(f"git failed: {exc}")
        logger.info("Manual fallback:")
        logger.info("  git clone --depth 1 https://github.com/CheyneyComputerScience/CREMA-D.git")
        logger.info(f"  Copy AudioWAV/ and VideoFlash/ into: {root}")
        raise

    return str(root)


def get_file_list(cremad_dir: str) -> List[Dict]:
    """
    Scan CREMA-D directory and return a metadata list for every valid sample.

    Filename schema  :  ``{ActorID}_{Sentence}_{Emotion}_{Level}.wav``
    Example          :  ``1001_DFA_ANG_XX.wav``

    Each dict contains:
        filename, audio_path, video_path (str | None), has_video,
        actor_id, sentence, emotion, level
    """
    root      = Path(cremad_dir).resolve()
    audio_dir = root / "AudioWAV"
    video_dir = root / "VideoFlash"

    if not audio_dir.exists():
        raise FileNotFoundError(
            f"AudioWAV/ not found in {root}. Run download_cremad() first."
        )

    samples: List[Dict] = []

    for wav in sorted(audio_dir.glob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) < 4:
            continue
        actor_id, sentence, emotion, level = parts[0], parts[1], parts[2], parts[3]
        if emotion not in VALID_EMOTIONS:
            continue

        flv = video_dir / f"{wav.stem}.flv"
        samples.append({
            "filename"  : wav.stem,
            "audio_path": str(wav),
            "video_path": str(flv) if flv.exists() else None,
            "has_video" : flv.exists(),
            "actor_id"  : actor_id,
            "sentence"  : sentence,
            "emotion"   : emotion,
            "level"     : level,
        })

    if not samples:
        raise RuntimeError(f"No valid samples found in {audio_dir}.")

    n_video = sum(s["has_video"] for s in samples)
    dist    = Counter(s["emotion"] for s in samples)

    logger.info(f"Loaded {len(samples)} samples  ({n_video} with video)")
    logger.info("Emotion distribution:")
    for emo in sorted(dist):
        bar = "█" * (dist[emo] // 50)
        logger.info(f"  {emo}  {dist[emo]:5d}  {bar}")

    return samples