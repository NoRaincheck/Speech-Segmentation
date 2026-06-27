"""Audio loading and resampling utilities."""

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def load_audio(path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.

    Returns:
        Tuple of (audio samples as float32, sample rate).
    """
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        num_samples = int(len(audio) * target_sr / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, num_samples),
            np.arange(len(audio)),
            audio,
        )
        sr = target_sr
    return audio, sr
