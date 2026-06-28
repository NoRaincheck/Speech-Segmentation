"""Optional VAE-based embedding enhancement for few-shot speaker matching.

Transforms ECAPA-TDNN embeddings into a structured latent space via a
variational bottleneck. Training uses three combined losses plus hard
negative mining to produce a discriminative latent space:

- **Reconstruction loss** (MSE): preserves input information.
- **Center loss** (Wen et al., ECCV 2016): pulls same-speaker embeddings
  toward their per-class center in latent space.
- **Margin loss**: pushes different-speaker centers apart via cosine
  similarity with a margin threshold.
- **Hard negative mining**: for each sample, finds the closest
  different-speaker center and applies extra margin pressure. This
  directly targets the hardest discrimination cases (similar speakers).
- **Prototype augmentation**: samples synthetic embeddings from a
  Gaussian centered on each class prototype during training. Expands
  the decision boundary without requiring additional audio data.
- **Learned diagonal metric**: a per-dimension weight vector trained
  alongside the VAE that amplifies discriminative latent dimensions
  and suppresses noisy ones. Applied before L2-normalization at
  inference time.

The latent vectors are L2-normalized at output so that cosine similarity
(dot product) can be used directly for matching.

Requires PyTorch (lazy-imported only when this module is used).
"""

from __future__ import annotations

import numpy as np
import torch


class SpeakerVAE:
    """Transforms ECAPA-TDNN embeddings via a variational bottleneck.

    The encoder maps embeddings to a learned latent space (mean + logvar),
    and the reparameterization trick produces the latent vector. At inference,
    only the encoder is needed. A learned diagonal metric scales latent
    dimensions before L2-normalization, amplifying discriminative features.

    The latent dimension and metric weights are stored in the checkpoint
    and loaded at init time (see ``_vae_config`` key in the ``.pt`` file).

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

        self._model = _VAE(self.INPUT_DIM, self.LATENT_DIM).to(self._device)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()

    def encode(self, embedding: np.ndarray) -> np.ndarray:
        """Encode a single embedding to the latent space.

        Applies the learned diagonal metric to scale dimensions, then
        L2-normalizes so cosine similarity equals dot product.

        Args:
            embedding: L2-normalized ECAPA-TDNN embedding (192-dim by default).

        Returns:
            L2-normalized latent vector (numpy, dim = ``self.LATENT_DIM``).
        """
        torch = self._torch
        t = torch.from_numpy(embedding).float().unsqueeze(0).to(self._device)
        with torch.no_grad():
            mu, _logvar = self._model.encode(t)
            z = mu * self._model._metric_scale
        z = z.squeeze(0).cpu().numpy()
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
            z = mu * self._model._metric_scale
        z = z.cpu().numpy()
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        z = z / norms
        return z


class _VAE(torch.nn.Module):  # type: ignore[name-defined]
    """Internal MLP-based VAE with learned diagonal metric.

    The ``_metric_scale`` parameter is a per-dimension positive weight
    (initialized to ones, clamped to [0.01, 10.0]) that amplifies
    discriminative latent dimensions. It is trained alongside the encoder
    and decoder, and applied after the encoder's mu layer at inference.
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        torch = self._get_torch()

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

        # Learned diagonal metric: per-dimension scaling for discriminability.
        # Initialized to ones; trained via backprop through the loss.
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

    @staticmethod
    def _get_torch():
        import torch

        return torch

    def encode(self, x):
        torch = self._get_torch()
        h = self._encoder(x)
        return self._fc_mu(h), self._fc_logvar(h)

    @staticmethod
    def _reparameterize(mu, logvar):
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
