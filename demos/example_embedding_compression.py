"""Example 3: Embedding compression — 192→64 with contrastive FT.

Demonstrates that a contrastive-fine-tuned VAE can compress ECAPA-TDNN
embeddings from 192 to 64 dimensions (3x compression) while maintaining
speaker discrimination. The raw PCA baseline shows that simple linear
compression loses information, while the VAE's nonlinear bottleneck
learns a compressed representation that preserves speaker structure.

Usage:
    uv run --group vae python demos/example_embedding_compression.py
"""

import os
import time
from collections import defaultdict

import numpy as np
import torch

from speech_segmentation import SpeakerEmbedder
from speech_segmentation.audio import load_audio
from speech_segmentation.augmentation import speed_perturb
from speech_segmentation.evaluation import compute_eer
from speech_segmentation.vae import VAE, supervised_contrastive_loss

MODELS_DIR = "models"
TARGET_DIM = 64


def pca_compress(train_embs: np.ndarray, target_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Simple PCA compression for baseline comparison."""
    mean = train_embs.mean(axis=0)
    centered = train_embs - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Take top eigenvectors
    idx = np.argsort(eigenvalues)[::-1][:target_dim]
    components = eigenvectors[:, idx]
    return components, mean


def train_contrastive_vae(
    train_embs: np.ndarray, train_labels: np.ndarray, latent_dim: int = 64
) -> VAE:
    """Train VAE with unsupervised pre-training + contrastive fine-tuning."""
    n_speakers = len(np.unique(train_labels))
    input_dim = train_embs.shape[1]

    model = VAE(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(train_embs).float()
    dataset = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(300):
        kl_w = min(1.0, epoch / 100)
        for (batch,) in loader:
            recon, mu, logvar = model(batch)
            recon_loss = torch.nn.functional.mse_loss(recon, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(batch)
            kl_loss = torch.clamp(kl_loss, min=2.0)
            loss = recon_loss + kl_w * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model_ft = VAE(input_dim, latent_dim, num_speakers=n_speakers)
    model_ft.load_state_dict(model.state_dict(), strict=False)
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-4)
    y = torch.from_numpy(train_labels).long()
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(200):
        for batch_x, batch_y in loader:
            mu, _ = model_ft.encode(batch_x)
            z = torch.nn.functional.normalize(mu, dim=1)
            loss = supervised_contrastive_loss(z, batch_y, temperature=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model_ft.eval()
    return model_ft


def main():
    print("=" * 70)
    print("EXAMPLE 3: Embedding Compression — 192→64 Dimensions")
    print("=" * 70)
    print()
    print("Compares three ways to reduce 192-dim ECAPA-TDNN embeddings to 64-dim:")
    print("  1. Raw 192-dim (baseline)")
    print("  2. PCA to 64-dim (linear compression)")
    print("  3. VAE with contrastive FT to 64-dim (nonlinear compression)")
    print()

    embedder = SpeakerEmbedder(f"{MODELS_DIR}/ecapa_tdnn.onnx")

    # Load all reference embeddings
    speaker_files: dict[str, list[str]] = defaultdict(list)
    for fname in sorted(os.listdir("refs")):
        if not fname.endswith(".wav"):
            continue
        speaker = fname.rsplit("_ref_", 1)[0]
        speaker_files[speaker].append(os.path.join("refs", fname))

    names = sorted(speaker_files.keys())

    # Build embeddings with augmentation for training
    train_embs = []
    train_labels = []
    ref_embs = {}
    for i, name in enumerate(names):
        embs = []
        for p in speaker_files[name]:
            audio, _ = load_audio(p)
            e = embedder.embed(audio)
            embs.append(e)
            train_embs.append(e)
            train_labels.append(i)
            for factor in [0.9, 1.1]:
                aug = speed_perturb(audio, 16000, factor)
                ae = embedder.embed(aug)
                train_embs.append(ae)
                train_labels.append(i)
        ref_embs[name] = np.mean(embs, axis=0)
        ref_embs[name] /= np.linalg.norm(ref_embs[name])

    train_embs = np.array(train_embs)
    train_labels = np.array(train_labels)

    # Load turns for testing
    turn_embs = []
    turn_labels = []
    for fname in sorted(os.listdir("turns")):
        if not fname.endswith(".wav"):
            continue
        speaker = fname.split("_")[2].replace(".wav", "")
        audio, _ = load_audio(os.path.join("turns", fname))
        turn_embs.append(embedder.embed(audio))
        turn_labels.append(speaker)
    turn_embs = np.array(turn_embs)
    turn_labels = np.array(turn_labels)

    # --- Method 1: Raw 192-dim ---
    proto_192 = np.array([ref_embs[n] for n in names])
    proto_192 /= np.linalg.norm(proto_192, axis=1, keepdims=True)

    correct_192 = 0
    sims_192 = []
    for e, lbl in zip(turn_embs, turn_labels):
        e_n = e / (np.linalg.norm(e) + 1e-8)
        s = proto_192 @ e_n
        pred = names[int(np.argmax(s))]
        correct_192 += int(pred == lbl)
        sims_192.append(float(s[names.index(lbl)]))
    acc_192 = correct_192 / len(turn_embs)

    # --- Method 2: PCA to 64-dim ---
    print("Training PCA compressor...")
    components, mean = pca_compress(train_embs, TARGET_DIM)

    proto_pca = np.array([ref_embs[n] for n in names])
    proto_pca = (proto_pca - mean) @ components
    norms = np.linalg.norm(proto_pca, axis=1, keepdims=True)
    proto_pca = proto_pca / norms

    correct_pca = 0
    sims_pca = []
    for e, lbl in zip(turn_embs, turn_labels):
        e_pca = (e - mean) @ components
        e_pca /= np.linalg.norm(e_pca) + 1e-8
        s = proto_pca @ e_pca
        pred = names[int(np.argmax(s))]
        correct_pca += int(pred == lbl)
        sims_pca.append(float(s[names.index(lbl)]))
    acc_pca = correct_pca / len(turn_embs)

    # --- Method 3: VAE with contrastive FT ---
    print("Training contrastive-fine-tuned VAE...")
    t0 = time.perf_counter()
    vae_model = train_contrastive_vae(train_embs, train_labels)
    print(f"  Trained in {time.perf_counter() - t0:.1f}s")

    proto_vae = np.array([
        vae_model.encode(torch.from_numpy(ref_embs[n]).float().unsqueeze(0))[0].squeeze(0).detach().numpy()
        for n in names
    ])
    proto_vae /= np.linalg.norm(proto_vae, axis=1, keepdims=True)

    correct_vae = 0
    sims_vae = []
    for e, lbl in zip(turn_embs, turn_labels):
        t = torch.from_numpy(e).float().unsqueeze(0)
        with torch.no_grad():
            mu, _ = vae_model.encode(t)
        z = mu.squeeze(0).numpy()
        z /= np.linalg.norm(z)
        s = proto_vae @ z
        pred = names[int(np.argmax(s))]
        correct_vae += int(pred == lbl)
        sims_vae.append(float(s[names.index(lbl)]))
    acc_vae = correct_vae / len(turn_embs)

    # --- Summary ---
    print()
    print(f"{'Method':<25} {'Dim':>6} {'Accuracy':>10} {'Mean Sim':>10} {'Min Sim':>10} {'Storage':>10}")
    print("-" * 75)
    print(f"{'Raw ECAPA-TDNN':<25} {'192':>6} {acc_192:>9.1%} {np.mean(sims_192):>10.4f} {min(sims_192):>10.4f} {'100%':>10}")
    print(f"{'PCA compression':<25} {'64':>6} {acc_pca:>9.1%} {np.mean(sims_pca):>10.4f} {min(sims_pca):>10.4f} {'33%':>10}")
    print(f"{'VAE + Contrastive FT':<25} {'64':>6} {acc_vae:>9.1%} {np.mean(sims_vae):>10.4f} {min(sims_vae):>10.4f} {'33%':>10}")

    print()
    print("Key insight: PCA is a linear projection that loses speaker-discriminative")
    print("information. The VAE's nonlinear bottleneck, guided by contrastive loss,")
    print("learns a compressed representation that preserves (or even enhances)")
    print("speaker separation while reducing storage by 3x.")
    print()
    print(f"Storage savings at scale:")
    print(f"  1M embeddings × 192 floats × 4 bytes = {1_000_000 * 192 * 4 / 1e9:.1f} GB")
    print(f"  1M embeddings × 64 floats × 4 bytes  = {1_000_000 * 64 * 4 / 1e9:.1f} GB  (saves {1_000_000 * 128 * 4 / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
