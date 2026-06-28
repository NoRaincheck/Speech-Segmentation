"""VAE-based embedding enhancement for few-shot speaker matching.

Transforms ECAPA-TDNN embeddings into a structured latent space via a
variational bottleneck. The encoder maps embeddings to a learned latent
space (mean + logvar), and the reparameterization trick produces the
latent vector. At inference, only the encoder is needed.

The latent vectors are L2-normalized at output so that cosine similarity
(dot product) can be used directly for matching.

Requires PyTorch (lazy-imported only when this module is used).
"""

from __future__ import annotations

import numpy as np
import torch


class SpeakerVAE:
    """Transforms ECAPA-TDNN embeddings via a variational bottleneck.

    Args:
        model_path: Path to saved ``.pt`` weights (``torch.save`` format).
            The checkpoint must contain ``_vae_config`` with ``input_dim``
            and ``latent_dim`` keys, plus the model ``state_dict``.
    """

    INPUT_DIM = 192

    def __init__(self, model_path: str) -> None:
        import torch

        self._torch = torch
        self._device = torch.device("cpu")

        checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)
        config = checkpoint["_vae_config"]
        self.INPUT_DIM = config["input_dim"]
        self.LATENT_DIM = config["latent_dim"]

        self._model = VAE(self.INPUT_DIM, self.LATENT_DIM).to(self._device)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()

    def encode(self, embedding: np.ndarray) -> np.ndarray:
        """Encode a single embedding to the latent space.

        Args:
            embedding: L2-normalized ECAPA-TDNN embedding (192-dim by default).

        Returns:
            L2-normalized latent vector (numpy, dim = ``self.LATENT_DIM``).
        """
        torch = self._torch
        t = torch.from_numpy(embedding).float().unsqueeze(0).to(self._device)
        with torch.no_grad():
            mu, _logvar = self._model.encode(t)
        z = mu.squeeze(0).cpu().numpy()
        z = z / np.linalg.norm(z)
        return z

    def encode_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Encode multiple embeddings at once.

        Args:
            embeddings: Array of shape ``(N, input_dim)``.

        Returns:
            L2-normalized array of shape ``(N, latent_dim)``.
        """
        torch = self._torch
        t = torch.from_numpy(embeddings).float().to(self._device)
        with torch.no_grad():
            mu, _logvar = self._model.encode(t)
        z = mu.cpu().numpy()
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        z = z / norms
        return z


class VAE(torch.nn.Module):  # type: ignore[name-defined]
    """MLP-based VAE used for training and inference."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        torch = self._get_torch()

        self._encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(0.2),
        )
        self._fc_mu = torch.nn.Linear(64, latent_dim)
        self._fc_logvar = torch.nn.Linear(64, latent_dim)

        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, input_dim),
        )

    @staticmethod
    def _get_torch():
        import torch

        return torch

    def encode(self, x):
        torch = self._get_torch()
        h = self._encoder(x)
        return self._fc_mu(h), self._fc_logvar(h)

    def _reparameterize(self, mu, logvar):
        torch = self._get_torch()
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self._decoder(z)

    def forward(self, x):
        torch = self._get_torch()
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
