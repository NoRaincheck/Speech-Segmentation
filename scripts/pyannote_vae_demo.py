"""
VAE on pyannote segmentation embeddings.

Trains a VAE on 128-dim embeddings extracted from the pyannote model's
penultimate layer. These embeddings capture speech characteristics from
the segmentation model's internal representation.

Two training modes:
  1. Unsupervised (reconstruction + KL on unlabelled data)
  2. Unsupervised + contrastive fine-tuning (if labelled refs available)

Demonstrates that the segmentation model's intermediate layers contain
useful speaker information that can be enhanced with a VAE bottleneck.

Usage:
    uv run --group vae python scripts/pyannote_vae_demo.py
"""

import os
import time

import numpy as np
import torch

from speech_segmentation.audio import load_audio
from speech_segmentation.augmentation import speed_perturb
from speech_segmentation.pyannote_embedding import PyannoteEmbedder
from speech_segmentation.vae import VAE, SpeakerVAE, supervised_contrastive_loss

MODELS_DIR = "models"


def load_unlabelled_embeddings(
    embedder: PyannoteEmbedder, unlabelled_dir: str
) -> np.ndarray:
    """Load and embed all unlabelled audio files with speed augmentation."""
    all_embs = []
    for fname in sorted(os.listdir(unlabelled_dir)):
        if not fname.endswith(".wav") or "_speed" in fname:
            continue
        audio, sr = load_audio(os.path.join(unlabelled_dir, fname))
        emb = embedder.embed(audio)
        all_embs.append(emb)
        for factor in [0.9, 1.1]:
            aug = speed_perturb(audio, sr, factor)
            all_embs.append(embedder.embed(aug))
    return np.array(all_embs)


def train_vae_unsupervised(
    embeddings: np.ndarray, latent_dim: int = 64, epochs: int = 500
) -> VAE:
    """Train VAE with reconstruction + KL loss only."""
    input_dim = embeddings.shape[1]
    model = VAE(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.from_numpy(embeddings).float()
    dataset = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        kl_weight = min(1.0, epoch / 100)
        for (batch,) in loader:
            recon, mu, logvar = model(batch)
            recon_loss = torch.nn.functional.mse_loss(recon, batch)
            kl_loss = (
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(batch)
            )
            kl_loss = torch.clamp(kl_loss, min=2.0)
            loss = recon_loss + kl_weight * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def train_vae_contrastive(
    embeddings: np.ndarray,
    labels: np.ndarray,
    latent_dim: int = 64,
    pretrain_epochs: int = 500,
    finetune_epochs: int = 200,
) -> VAE:
    """Unsupervised pre-train + contrastive fine-tune."""
    n_speakers = len(np.unique(labels))
    base = train_vae_unsupervised(embeddings, latent_dim, pretrain_epochs)

    model = VAE(embeddings.shape[1], latent_dim, num_speakers=n_speakers)
    model.load_state_dict(base.state_dict(), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(labels).long()
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(finetune_epochs):
        for batch_x, batch_y in loader:
            mu, _ = model.encode(batch_x)
            z = torch.nn.functional.normalize(mu, dim=1)
            loss = supervised_contrastive_loss(z, batch_y, temperature=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def compute_separation(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine similarity (lower = more spread out)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)
    sim_matrix = normed @ normed.T
    np.fill_diagonal(sim_matrix, 0)
    n = len(embeddings)
    return float(sim_matrix.sum()) / (n * (n - 1))


def main():
    print("=" * 70)
    print("VAE ON PYANNOTE SEGMENTATION EMBEDDINGS")
    print("=" * 70)
    print()

    embedder_path = f"{MODELS_DIR}/model_with_embedding.onnx"
    if not os.path.exists(embedder_path):
        print(f"ERROR: {embedder_path} not found.")
        print(
            "Run: uv run --group export python scripts/pyannote_embedding_export.py"
        )
        return

    embedder = PyannoteEmbedder(embedder_path)

    print("Loading unlabelled embeddings...")
    t0 = time.perf_counter()
    unlabelled_embs = load_unlabelled_embeddings(embedder, "unlabelled")
    print(
        f"  {len(unlabelled_embs)} embeddings (128-dim) in {time.perf_counter() - t0:.1f}s"
    )

    print("\nTraining unsupervised VAE (128 -> 64 latent)...")
    t0 = time.perf_counter()
    vae_model = train_vae_unsupervised(unlabelled_embs, latent_dim=64, epochs=500)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    config = {"input_dim": 128, "latent_dim": 64}
    checkpoint = {"_vae_config": config, "state_dict": vae_model.state_dict()}
    ckpt_path = f"{MODELS_DIR}/pyannote_vae.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    vae = SpeakerVAE(ckpt_path)
    projected = vae.encode_batch(unlabelled_embs)

    raw_sep = compute_separation(unlabelled_embs)
    proj_sep = compute_separation(projected)

    print(f"\nEmbedding space comparison:")
    print(f"  Raw pyannote (128-dim):  mean cosine sim = {raw_sep:.4f}")
    print(f"  VAE-projected (64-dim):  mean cosine sim = {proj_sep:.4f}")
    print(f"  Separation improvement:  {raw_sep - proj_sep:+.4f}")

    if not os.path.isdir("refs") or not os.path.isdir("turns"):
        print("\nNeed refs/ and turns/ directories for classification test.")
        print("Done.")
        return

    from collections import defaultdict

    # Build speaker prototypes from reference clips
    ref_embs: dict[str, list[np.ndarray]] = defaultdict(list)
    for fname in sorted(os.listdir("refs")):
        if not fname.endswith(".wav"):
            continue
        speaker = fname.rsplit("_ref_", 1)[0]
        audio, _ = load_audio(os.path.join("refs", fname))
        ref_embs[speaker].append(embedder.embed(audio))

    names = sorted(ref_embs.keys())
    raw_proto = np.array([np.mean(ref_embs[n], axis=0) for n in names])
    raw_proto /= np.linalg.norm(raw_proto, axis=1, keepdims=True)

    vae_proto = vae.encode_batch(raw_proto)

    # Load test turns
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

    # Classify: raw pyannote
    correct_raw = 0
    for e, lbl in zip(turn_embs, turn_labels):
        e_n = e / (np.linalg.norm(e) + 1e-8)
        pred = names[int(np.argmax(raw_proto @ e_n))]
        correct_raw += int(pred == lbl)
    acc_raw = correct_raw / len(turn_embs)

    # Classify: VAE-projected
    vae_turns = vae.encode_batch(turn_embs)
    correct_vae = 0
    for e, lbl in zip(vae_turns, turn_labels):
        pred = names[int(np.argmax(vae_proto @ e))]
        correct_vae += int(pred == lbl)
    acc_vae = correct_vae / len(turn_embs)

    raw_proto_sep = compute_separation(raw_proto)
    vae_proto_sep = compute_separation(vae_proto)

    print()
    print(f"{'Method':<25} {'Dim':>6} {'Accuracy':>10} {'Proto Sep':>10}")
    print("-" * 55)
    print(f"{'Raw pyannote':<25} {'128':>6} {acc_raw:>9.1%} {raw_proto_sep:>10.4f}")
    print(f"{'VAE-projected':<25} {'64':>6} {acc_vae:>9.1%} {vae_proto_sep:>10.4f}")
    print(f"{'Delta':<25} {'':>6} {acc_vae - acc_raw:>+9.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
