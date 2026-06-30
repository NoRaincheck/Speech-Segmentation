"""Example 1: Domain mismatch — the VAE as a domain-invariant projector.

In practice, ECAPA-TDNN was trained on VoxCeleb (clean studio). When
deployed on telephony/meetings, accuracy drops. A domain-aware VAE can
help — but only when the domain shift is significant enough to break
the baseline.

This example shows the boundary: at what corruption level does the VAE
start winning? We use aggressive corruption (dimension dropout + bias +
gain) to find that boundary.

Usage:
    uv run --group vae python demos/example_domain_mismatch.py
"""

import os
import time
from collections import defaultdict

import numpy as np
import torch

from speech_segmentation import SpeakerEmbedder
from speech_segmentation.audio import load_audio
from speech_segmentation.augmentation import speed_perturb
from speech_segmentation.vae import VAE, supervised_contrastive_loss

MODELS_DIR = "models"
LATENT_DIM = 64
INPUT_DIM = 192


def corrupt_embeddings(embs: np.ndarray, severity: float, rng: np.random.RandomState) -> np.ndarray:
    """Apply channel corruption: dimension dropout + bias + gain."""
    dim = embs.shape[1]
    corrupted = embs.copy()
    n_drop = int(severity * dim * 0.5)
    if n_drop > 0:
        drop_mask = rng.choice(dim, n_drop, replace=False)
        corrupted[:, drop_mask] = 0.0
    bias = severity * 0.5 * rng.randn(dim)
    corrupted = corrupted + bias
    gain = 1.0 + severity * 0.8 * rng.randn(dim)
    corrupted = corrupted * gain
    norms = np.linalg.norm(corrupted, axis=1, keepdims=True)
    return corrupted / (norms + 1e-8)


def train_vae(clean_embs, labels, rng):
    n_speakers = len(np.unique(labels))
    model = VAE(INPUT_DIM, LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(clean_embs).float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=32, shuffle=True)

    for epoch in range(400):
        kl_w = min(1.0, epoch / 100)
        for (batch,) in loader:
            sev = rng.uniform(0.1, 0.5)
            corrupted = corrupt_embeddings(batch.numpy(), sev, rng)
            recon, mu, logvar = model(torch.from_numpy(corrupted).float())
            loss = torch.nn.functional.mse_loss(recon, batch) + kl_w * torch.clamp(
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(batch), min=2.0
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model_ft = VAE(INPUT_DIM, LATENT_DIM, num_speakers=n_speakers)
    model_ft.load_state_dict(model.state_dict(), strict=False)
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-4)
    y = torch.from_numpy(labels).long()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=32, shuffle=True)
    for epoch in range(200):
        for bx, by in loader:
            mu, _ = model_ft.encode(bx)
            loss = supervised_contrastive_loss(torch.nn.functional.normalize(mu, dim=1), by, 0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model_ft.eval()
    return model_ft


def classify(proto, turns, names):
    c = 0
    for spk, embs in turns.items():
        for e in embs:
            if names[int(np.argmax(proto @ e))] == spk:
                c += 1
    return c / sum(len(v) for v in turns.values())


def classify_vae(vae, proto, turns, names):
    c = 0
    for spk, embs in turns.items():
        for e in embs:
            t = torch.from_numpy(e).float().unsqueeze(0)
            with torch.no_grad():
                z = vae.encode(t)[0].squeeze(0).numpy()
            z /= np.linalg.norm(z)
            if names[int(np.argmax(proto @ z))] == spk:
                c += 1
    return c / sum(len(v) for v in turns.values())


def main():
    print("=" * 70)
    print("EXAMPLE 1: Domain Mismatch — Finding the VAE Advantage Boundary")
    print("=" * 70)
    print()
    print("At what corruption level does VAE start helping?")
    print("We corrupt test embeddings (dimension dropout + bias + gain)")
    print("and measure baseline vs VAE accuracy at each level.")
    print()

    embedder = SpeakerEmbedder(f"{MODELS_DIR}/ecapa_tdnn.onnx")
    rng = np.random.RandomState(42)

    speaker_files = defaultdict(list)
    for fname in sorted(os.listdir("refs")):
        if fname.endswith(".wav"):
            speaker_files[fname.rsplit("_ref_", 1)[0]].append(os.path.join("refs", fname))
    names = sorted(speaker_files.keys())

    ref_embs = {}
    train_e, train_l = [], []
    for i, name in enumerate(names):
        embs = []
        for p in speaker_files[name]:
            audio, _ = load_audio(p)
            e = embedder.embed(audio)
            embs.append(e)
            train_e.append(e)
            train_l.append(i)
            for f in [0.9, 1.1]:
                train_e.append(embedder.embed(speed_perturb(audio, 16000, f)))
                train_l.append(i)
        ref_embs[name] = np.mean(embs, axis=0)
        ref_embs[name] /= np.linalg.norm(ref_embs[name])

    clean_proto = np.array([ref_embs[n] for n in names])
    clean_proto /= np.linalg.norm(clean_proto, axis=1, keepdims=True)

    turn_embs = {}
    for fname in sorted(os.listdir("turns")):
        if fname.endswith(".wav"):
            spk = fname.split("_")[2].replace(".wav", "")
            audio, _ = load_audio(os.path.join("turns", fname))
            turn_embs.setdefault(spk, []).append(embedder.embed(audio))

    print("Training VAE...")
    vae = train_vae(np.array(train_e), np.array(train_l), np.random.RandomState(42))
    vae_proto = np.array([
        vae.encode(torch.from_numpy(ref_embs[n]).float().unsqueeze(0))[0].squeeze(0).detach().numpy()
        for n in names
    ])
    vae_proto /= np.linalg.norm(vae_proto, axis=1, keepdims=True)

    print()
    print(f"{'Severity':<12} {'Dims dropped':>14} {'Baseline':>10} {'VAE':>10} {'VAE wins?':>10}")
    print("-" * 60)

    for sev in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        rng_e = np.random.RandomState(99)
        corrupted = {
            spk: [corrupt_embeddings(e[np.newaxis], sev, rng_e)[0] for e in embs]
            for spk, embs in turn_embs.items()
        }
        n_drop = int(sev * INPUT_DIM * 0.5)
        acc_b = classify(clean_proto, corrupted, names)
        acc_v = classify_vae(vae, vae_proto, corrupted, names)
        wins = "YES" if acc_v > acc_b else ("tie" if acc_v == acc_b else "no")
        print(f"  {sev:<10.1f} {n_drop:>10}/{INPUT_DIM} {acc_b:>9.1%} {acc_v:>9.1%} {wins:>10}")

    print()
    print("Key insight: ECAPA-TDNN is remarkably robust to moderate corruption.")
    print("The VAE bottleneck only helps when corruption is severe enough to")
    print("break the baseline (typically >30% dimension loss). In practice,")
    print("this corresponds to extreme domain shift (e.g., very low bitrate")
    print("codec, severe hardware mismatch). For mild-moderate shift, the")
    print("raw ECAPA-TDNN embeddings are sufficient.")


if __name__ == "__main__":
    main()
