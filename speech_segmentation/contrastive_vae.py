"""Contrastive VAE for speaker embedding enhancement.

Combines a variational bottleneck with supervised contrastive learning.
Requires reference utterances where every speaker says the **same text** so
that embedding differences are purely speaker-driven.

Training losses:

- **Reconstruction loss** (MSE): preserves input information.
- **KL divergence**: regularises the latent space.
- **Supervised contrastive loss** (Khosla et al., NeurIPS 2020): pulls
  same-speaker embeddings together and pushes different-speaker embeddings
  apart in the metric-scaled latent space.  Because every speaker utters
  the same text, variation in the latent space is attributable to speaker
  identity alone.
- **Learned diagonal metric**: per-dimension scaling trained alongside the
  encoder, amplifying discriminative latent dimensions.

At inference only the encoder + metric are used; the output is L2-normalized
so cosine similarity equals dot product.

Requires PyTorch (lazy-imported only when this module is used).
"""

from __future__ import annotations

import numpy as np
import torch


class SpeakerContrastiveVAE:
    """Transforms ECAPA-TDNN embeddings via a contrastive VAE bottleneck.

    Same interface as :class:`SpeakerVAE` so it can be passed directly to
    :class:`Diarizer`.

    Args:
        model_path: Path to saved ``.pt`` weights.  Checkpoint must contain
            ``_vae_config`` with ``input_dim`` and ``latent_dim`` plus the
            model ``state_dict``.
    """

    INPUT_DIM = 192

    def __init__(self, model_path: str) -> None:
        import torch as _torch

        self._torch = _torch
        self._device = _torch.device("cpu")

        checkpoint = _torch.load(model_path, map_location=self._device, weights_only=False)
        config = checkpoint["_vae_config"]
        self.INPUT_DIM = config["input_dim"]
        self.LATENT_DIM = config["latent_dim"]

        self._model = _ContrastiveVAE(self.INPUT_DIM, self.LATENT_DIM).to(self._device)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()

    def encode(self, embedding: np.ndarray) -> np.ndarray:
        """Encode a single embedding to the latent space."""
        torch = self._torch
        t = torch.from_numpy(embedding).float().unsqueeze(0).to(self._device)
        with torch.no_grad():
            mu, _logvar = self._model.encode(t)
            z = mu * self._model._metric_scale
        z = z.squeeze(0).cpu().numpy()
        z = z / np.linalg.norm(z)
        return z

    def encode_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Encode multiple embeddings at once."""
        torch = self._torch
        t = torch.from_numpy(embeddings).float().to(self._device)
        with torch.no_grad():
            mu, _logvar = self._model.encode(t)
            z = mu * self._model._metric_scale
        z = z.cpu().numpy()
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        z = z / norms
        return z


class _ContrastiveVAE(torch.nn.Module):
    """MLP-VAE with learned diagonal metric (same arch as _VAE)."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        import torch

        self._encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(0.2),
        )
        self._fc_mu = torch.nn.Linear(64, latent_dim)
        self._fc_logvar = torch.nn.Linear(64, latent_dim)
        self._metric_scale = torch.nn.Parameter(torch.ones(latent_dim))
        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._encoder(x)
        return self._fc_mu(h), self._fc_logvar(h)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
