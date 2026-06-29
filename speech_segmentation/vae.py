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
import torch.nn.functional as F


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
        num_speakers = config.get("num_speakers")

        self._model = VAE(self.INPUT_DIM, self.LATENT_DIM, num_speakers=num_speakers).to(
            self._device
        )
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
    """MLP-based VAE used for training and inference.

    Args:
        input_dim: Dimensionality of input embeddings.
        latent_dim: Dimensionality of the latent space.
        num_speakers: When provided, adds a linear classification head on
            the latent space for speaker discrimination. The ``classify``
            method returns logits over speaker classes.
    """

    def __init__(self, input_dim: int, latent_dim: int, num_speakers: int | None = None) -> None:
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

        self._classifier = (
            torch.nn.Linear(latent_dim, num_speakers) if num_speakers is not None else None
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

    def classify(self, z):
        """Classify speaker from a latent vector.

        Args:
            z: Latent tensor of shape ``(batch, latent_dim)``.

        Returns:
            Logits tensor of shape ``(batch, num_speakers)``.
            Returns None if no classification head was configured.
        """
        if self._classifier is None:
            return None
        return self._classifier(z)

    def forward(self, x):
        torch = self._get_torch()
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute supervised contrastive loss (Wu et al. 2018, CVPR).

    Pulls same-speaker embeddings together and pushes different-speaker
    embeddings apart in the latent space.

    Args:
        features: L2-normalized feature tensor of shape (batch, latent_dim).
        labels: Speaker label tensor of shape (batch,).
        temperature: Temperature scaling factor. Lower = harder contrasts.

    Returns:
        Scalar loss tensor.
    """
    device = features.device
    batch_size = features.shape[0]

    if batch_size <= 1:
        return torch.tensor(0.0, device=device)

    features = features / (features.norm(dim=1, keepdim=True) + 1e-8)

    sim_matrix = torch.matmul(features, features.T) / temperature

    labels = labels.unsqueeze(0)
    mask = (labels == labels.T).float().to(device)

    logits_mask = torch.ones_like(sim_matrix) - torch.eye(batch_size, device=device)
    sim_matrix = sim_matrix * logits_mask

    sim_matrix = torch.clamp(sim_matrix, min=-10.0, max=10.0)
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    mask_sum = mask.sum(dim=1)
    mean_log_prob = (mask * log_prob).sum(dim=1) / (mask_sum + 1e-8)

    loss = -mean_log_prob[mask_sum > 0].mean()
    return loss


def prototypical_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_speakers: int,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute prototypical network loss (Snell et al. 2017, ICML).

    Optimizes for prototype-based classification in the latent space.

    Args:
        features: L2-normalized feature tensor of shape (batch, latent_dim).
        labels: Speaker label tensor of shape (batch,).
        n_speakers: Number of unique speakers.
        temperature: Temperature for softmax. Lower = sharper predictions.

    Returns:
        Scalar loss tensor.
    """
    device = features.device

    prototypes = torch.zeros(n_speakers, features.shape[1], device=device)
    counts = torch.zeros(n_speakers, device=device)

    for i in range(n_speakers):
        mask = labels == i
        if mask.sum() > 0:
            prototypes[i] = features[mask].mean(dim=0)
            counts[i] = mask.sum()

    prototypes = F.normalize(prototypes, dim=1)

    valid_mask = counts > 0
    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=device)

    logits = torch.matmul(features, prototypes[valid_mask].T) / temperature
    targets = torch.zeros(len(features), dtype=torch.long, device=device)

    speaker_to_idx = {s.item(): i for i, s in enumerate(valid_mask.nonzero().squeeze(1))}
    for i, label in enumerate(labels):
        targets[i] = speaker_to_idx[label.item()]

    loss = F.cross_entropy(logits, targets)
    return loss
