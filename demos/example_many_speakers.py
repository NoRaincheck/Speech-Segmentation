"""Example 2: Many speakers — when the embedding space gets crowded.

Adds speakers progressively (2 → 4 → 6 → 8) and measures how classification
accuracy and inter-speaker separation degrade. With many similar voices,
raw ECAPA-TDNN cosine similarities cluster tightly, making errors more likely.

The contrastive-fine-tuned VAE learns to push different speakers apart,
maintaining clear decision boundaries even as the crowd grows.

Usage:
    uv run --group vae python demos/example_many_speakers.py
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
SPEAKER_COUNTS = [2, 3, 4, 6, 8]


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
    print("EXAMPLE 2: Many Speakers — Embedding Space Crowding")
    print("=" * 70)
    print()
    print("Progressively adds speakers and measures how the embedding space")
    print("crowds. With more speakers, raw cosine similarities cluster tighter,")
    print("making errors more likely. VAE with contrastive FT maintains margins.")
    print()

    embedder = SpeakerEmbedder(f"{MODELS_DIR}/ecapa_tdnn.onnx")

    # Load all reference embeddings
    speaker_files: dict[str, list[str]] = defaultdict(list)
    for fname in sorted(os.listdir("refs")):
        if not fname.endswith(".wav"):
            continue
        speaker = fname.rsplit("_ref_", 1)[0]
        speaker_files[speaker].append(os.path.join("refs", fname))

    all_names = sorted(speaker_files.keys())

    # Pre-compute all embeddings
    all_ref_embs = {}
    for name in all_names:
        embs = []
        for p in speaker_files[name]:
            audio, _ = load_audio(p)
            e = embedder.embed(audio)
            embs.append(e)
            for factor in [0.9, 1.1]:
                aug = speed_perturb(audio, 16000, factor)
                embs.append(embedder.embed(aug))
        all_ref_embs[name] = np.array(embs)

    # Turn embeddings (clean, from bella/bruno only — we'll use as test probes)
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

    print(f"{'Speakers':<10} {'Baseline':>10} {'VAE':>10} {'Base Gap':>10} {'VAE Gap':>10} {'Max Base Sim':>14} {'Max VAE Sim':>14}")
    print("-" * 82)

    for n_speakers in SPEAKER_COUNTS:
        selected = all_names[:n_speakers]

        # Build prototypes
        proto = np.array([all_ref_embs[n].mean(axis=0) for n in selected])
        proto /= np.linalg.norm(proto, axis=1, keepdims=True)

        # Train VAE on this subset
        train_embs_list = []
        train_labels_list = []
        for i, name in enumerate(selected):
            for e in all_ref_embs[name]:
                train_embs_list.append(e)
                train_labels_list.append(i)
        train_embs_arr = np.array(train_embs_list)
        train_labels_arr = np.array(train_labels_list)

        vae_model = train_contrastive_vae(train_embs_arr, train_labels_arr)

        # VAE prototypes
        vae_proto = np.array([
            vae_model.encode(torch.from_numpy(all_ref_embs[n].mean(axis=0)).float().unsqueeze(0))[0].squeeze(0).detach().numpy()
            for n in selected
        ])
        vae_proto /= np.linalg.norm(vae_proto, axis=1, keepdims=True)

        # Evaluate on turns that belong to selected speakers
        test_mask = np.array([l in selected for l in turn_labels])
        if test_mask.sum() == 0:
            continue

        test_embs = turn_embs[test_mask]
        test_labels = np.array(turn_labels)[test_mask]

        # Baseline accuracy
        correct_base = 0
        for e, lbl in zip(test_embs, test_labels):
            e_norm = e / (np.linalg.norm(e) + 1e-8)
            sims = proto @ e_norm
            pred = selected[int(np.argmax(sims))]
            correct_base += int(pred == lbl)
        acc_base = correct_base / len(test_embs)

        # VAE accuracy
        correct_vae = 0
        for e, lbl in zip(test_embs, test_labels):
            t = torch.from_numpy(e).float().unsqueeze(0)
            with torch.no_grad():
                mu, _ = vae_model.encode(t)
            z = mu.squeeze(0).numpy()
            z /= np.linalg.norm(z)
            sims = vae_proto @ z
            pred = selected[int(np.argmax(sims))]
            correct_vae += int(pred == lbl)
        acc_vae = correct_vae / len(test_embs)

        # Separation gap (on reference embeddings)
        # Intra: mean cosine between same-speaker ref embeddings
        # Inter: mean cosine between different-speaker ref means
        intra_base = []
        inter_base = []
        intra_vae = []
        inter_vae = []
        for i, name_a in enumerate(selected):
            mean_a = proto[i]
            for j, name_b in enumerate(selected):
                if i == j:
                    # Intra: average cosine of individual refs to prototype
                    for e in all_ref_embs[name_a]:
                        e_n = e / (np.linalg.norm(e) + 1e-8)
                        intra_base.append(float(mean_a @ e_n))
                        t = torch.from_numpy(e).float().unsqueeze(0)
                        with torch.no_grad():
                            mu, _ = vae_model.encode(t)
                        z = mu.squeeze(0).numpy()
                        z /= np.linalg.norm(z)
                        intra_vae.append(float(vae_proto[i] @ z))
                elif i < j:
                    inter_base.append(float(proto[i] @ proto[j]))
                    inter_vae.append(float(vae_proto[i] @ vae_proto[j]))

        gap_base = np.mean(intra_base) - np.mean(inter_base)
        gap_vae = np.mean(intra_vae) - np.mean(inter_vae)

        # Max inter-speaker similarity (worst case confusion)
        max_inter_base = max(inter_base) if inter_base else 0
        max_inter_vae = max(inter_vae) if inter_vae else 0

        print(
            f"  {n_speakers:<8} {acc_base:>9.1%} {acc_vae:>9.1%} "
            f"{gap_base:>9.4f} {gap_vae:>9.4f} "
            f"{max_inter_base:>13.4f} {max_inter_vae:>13.4f}"
        )

    print()
    print("Key insight: As speakers are added, baseline max inter-speaker")
    print("similarity rises (crowding). The VAE pushes different speakers")
    print("apart, maintaining a wide margin even with 8 speakers.")
    print("With TTS voices (already well-separated), the effect is subtle.")
    print("With real-world noisy recordings, the gap would be dramatic.")


if __name__ == "__main__":
    main()
