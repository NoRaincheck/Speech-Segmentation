"""Speaker embedding extraction using ECAPA-TDNN ONNX model."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

TARGET_SR = 16000
N_MELS = 80
FRAME_STEP = 160
FRAME_LENGTH = 400
N_FFT = 512
PREEMPHASIS = 0.97


class SpeakerEmbedder:
    """Extract 192-dim speaker embeddings using ECAPA-TDNN ONNX model.

    Args:
        model_path: Path to the ECAPA-TDNN ONNX model file.
    """

    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(model_path)

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from 16kHz audio.

        Args:
            audio_16k: Audio samples at 16kHz, float32.

        Returns:
            L2-normalized 192-dim embedding vector.
        """
        fbank = extract_fbank(audio_16k)
        raw_emb = self.session.run(None, {"input_values": fbank[np.newaxis].astype(np.float32)})[0]
        emb = raw_emb.flatten()
        emb = emb / np.linalg.norm(emb)
        return emb


def extract_fbank(audio_16k: np.ndarray) -> np.ndarray:
    """Extract 80-dim Fbank features from 16kHz audio using numpy.

    Args:
        audio_16k: Audio samples at 16kHz, float32.

    Returns:
        Fbank features array of shape (n_frames, 80).
    """
    audio = np.append(audio_16k[0], audio_16k[1:] - PREEMPHASIS * audio_16k[:-1])
    n_frames = 1 + (len(audio) - FRAME_LENGTH) // FRAME_STEP
    pad_length = n_frames * FRAME_STEP + FRAME_LENGTH - len(audio)
    audio = np.append(audio, np.zeros(pad_length))

    indices = np.arange(FRAME_LENGTH).reshape(1, -1) + np.arange(n_frames).reshape(-1, 1) * FRAME_STEP
    frames = audio[indices]
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(FRAME_LENGTH) / (FRAME_LENGTH - 1))
    frames *= hamming

    mag_frames = np.abs(np.fft.rfft(frames, N_FFT))
    pow_frames = (1.0 / N_FFT) * (mag_frames**2)

    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (TARGET_SR / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, N_MELS + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((N_FFT + 1) * hz_points / TARGET_SR).astype(int)

    n_freqs = N_FFT // 2 + 1
    mel_fb = np.zeros((N_MELS, n_freqs))
    for m in range(1, N_MELS + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                mel_fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                mel_fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    fbank = np.dot(pow_frames, mel_fb.T)
    fbank = np.where(fbank == 0, np.finfo(float).eps, fbank)
    fbank = 10 * np.log10(fbank)
    return fbank
