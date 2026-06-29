"""Data augmentation utilities for speech and speaker embeddings.

Provides speed perturbation, noise augmentation, and embedding-level
augmentation to improve VAE training with small datasets.
"""

from __future__ import annotations

import numpy as np


def speed_perturb(audio: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Resample audio to simulate speed change without pitch shift.

    Args:
        audio: Input audio samples.
        sr: Sample rate.
        factor: Speed factor (>1.0 = faster, <1.0 = slower).

    Returns:
        Resampled audio at the original sample rate.
    """
    indices = np.linspace(0, len(audio) - 1, int(len(audio) / factor))
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def add_noise(audio: np.ndarray, std: float) -> np.ndarray:
    """Add Gaussian noise to audio.

    Args:
        audio: Input audio samples.
        std: Standard deviation of noise.

    Returns:
        Noisy audio.
    """
    noise = np.random.randn(len(audio)).astype(np.float32) * std
    return audio + noise


def augment_embeddings(
    embeddings: np.ndarray,
    noise_stds: list[float] | None = None,
    n_copies: int = 1,
) -> np.ndarray:
    """Apply noise augmentation to embeddings.

    Args:
        embeddings: Array of shape (N, dim).
        noise_stds: List of noise standard deviations to apply.
            Defaults to [0.01, 0.03, 0.05].
        n_copies: Number of augmented copies per original.

    Returns:
        Augmented array of shape (N * (1 + n_copies * len(noise_stds)), dim).
    """
    if noise_stds is None:
        noise_stds = [0.01, 0.03, 0.05]

    augmented = [embeddings]
    for _ in range(n_copies):
        for std in noise_stds:
            noise = np.random.randn(*embeddings.shape).astype(np.float32) * std
            augmented.append(embeddings + noise)

    return np.concatenate(augmented, axis=0)


def generate_augmented_audio(
    audio: np.ndarray,
    sr: int,
    speed_factors: list[float] | None = None,
    noise_stds: list[float] | None = None,
) -> list[np.ndarray]:
    """Generate augmented versions of audio.

    Args:
        audio: Input audio samples.
        sr: Sample rate.
        speed_factors: Speed perturbation factors. Defaults to [0.9, 1.0, 1.1].
        noise_stds: Noise levels to add. Defaults to [0.005, 0.01].

    Returns:
        List of augmented audio arrays (including original).
    """
    if speed_factors is None:
        speed_factors = [0.9, 1.0, 1.1]
    if noise_stds is None:
        noise_stds = [0.005, 0.01]

    augmented = [audio]
    for factor in speed_factors:
        if factor != 1.0:
            augmented.append(speed_perturb(audio, sr, factor))
    for std in noise_stds:
        augmented.append(add_noise(audio, std))

    return augmented
