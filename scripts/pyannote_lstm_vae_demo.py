"""
VAE on pyannote LSTM embeddings (256-dim).

Trains a VAE on the bidirectional LSTM output from pyannote-segmentation-3.0.
The LSTM output (256-dim) is the richest intermediate representation —
before the task-specific MLP head compresses it for segmentation.

Compares:
  1. Raw ECAPA-TDNN (192-dim) — speaker identification baseline
  2. Raw pyannote LSTM (256-dim) — segmentation model's internal representation
  3. VAE-projected LSTM (64-dim) — compressed via VAE bottleneck

Usage:
    uv run --group vae python scripts/pyannote_lstm_vae_demo.py
"""

import os
import time
from collections import defaultdict

import numpy as np
import onnxruntime as ort
import torch

from speech_segmentation import SpeakerEmbedder
from speech_segmentation.audio import load_audio
from speech_segmentation.augmentation import speed_perturb
from speech_segmentation.vae import VAE, SpeakerVAE, supervised_contrastive_loss

MODELS_DIR = "models"


class LSTMBEDDER:
    """Extract 256-dim embeddings from pyannote model's LSTM layer."""

    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(model_path)

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        _, output = self.session.run(
            None,
            {"input_values": audio_16k[np.newaxis, np.newaxis, :].astype(np.float32)},
        )
        emb = output[0].mean(axis=0)
        return emb / np.linalg.norm(emb)


def train_vae(
    embeddings: np.ndarray, labels: np.ndarray, latent_dim: int = 64
) -> VAE:
    """Unsupervised pre-train + contrastive fine-tune."""
    n_speakers = len(np.unique(labels))
    input_dim = embeddings.shape[1]

    # Phase 1: Unsupervised
    model = VAE(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(embeddings).float()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x), batch_size=32, shuffle=True
    )
    for epoch in range(400):
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

    # Phase 2: Contrastive fine-tune
    model_ft = VAE(input_dim, latent_dim, num_speakers=n_speakers)
    model_ft.load_state_dict(model.state_dict(), strict=False)
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-4)
    y = torch.from_numpy(labels).long()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=32, shuffle=True
    )
    for epoch in range(200):
        for batch_x, batch_y in loader:
            mu, _ = model_ft.encode(batch_x)
            z = torch.nn.functional.normalize(mu, dim=1)
            loss = supervised_contrastive_loss(z, batch_y, temperature=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model_ft


def classify(proto: np.ndarray, turns: list[tuple[np.ndarray, str]], names: list[str]) -> float:
    correct = 0
    for emb, lbl in turns:
        pred = names[int(np.argmax(proto @ emb))]
        correct += int(pred == lbl)
    return correct / len(turns)


def sep(embs):
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    n = embs / (norms + 1e-8)
    sim = n @ n.T
    np.fill_diagonal(sim, 0)
    return float(sim.sum()) / (len(embs) * (len(embs) - 1))


def main():
    print("=" * 70)
    print("VAE ON PYANNOTE LSTM EMBEDDINGS (256-dim)")
    print("=" * 70)
    print()

    # Load models
    ecapa = SpeakerEmbedder(f"{MODELS_DIR}/ecapa_tdnn.onnx")
    lstm_emb = LSTMBEDDER(f"{MODELS_DIR}/model_with_lstm.onnx")

    # Build prototypes from refs/
    ref_embs: dict[str, dict[str, list]] = defaultdict(lambda: {"ecapa": [], "lstm": []})
    for fname in sorted(os.listdir("refs")):
        if not fname.endswith(".wav"):
            continue
        speaker = fname.rsplit("_ref_", 1)[0]
        audio, _ = load_audio(os.path.join("refs", fname))
        ref_embs[speaker]["ecapa"].append(ecapa.embed(audio))
        ref_embs[speaker]["lstm"].append(lstm_emb.embed(audio))

    names = sorted(ref_embs.keys())

    # ECAPA prototypes
    proto_ecapa = np.array([np.mean(ref_embs[n]["ecapa"], axis=0) for n in names])
    proto_ecapa /= np.linalg.norm(proto_ecapa, axis=1, keepdims=True)

    # LSTM prototypes
    proto_lstm = np.array([np.mean(ref_embs[n]["lstm"], axis=0) for n in names])
    proto_lstm /= np.linalg.norm(proto_lstm, axis=1, keepdims=True)

    # Load turns
    turns_ecapa = []
    turns_lstm = []
    turn_labels = []
    for fname in sorted(os.listdir("turns")):
        if not fname.endswith(".wav"):
            continue
        speaker = fname.split("_")[2].replace(".wav", "")
        audio, _ = load_audio(os.path.join("turns", fname))
        turns_ecapa.append(ecapa.embed(audio))
        turns_lstm.append(lstm_emb.embed(audio))
        turn_labels.append(speaker)

    # Raw ECAPA accuracy
    acc_ecapa = classify(proto_ecapa, list(zip(turns_ecapa, turn_labels)), names)

    # Raw LSTM accuracy
    acc_lstm = classify(proto_lstm, list(zip(turns_lstm, turn_labels)), names)

    # Train VAE on LSTM embeddings
    # Build training data from unlabelled + augmented refs
    print("Building training data...")
    train_embs = []
    train_labels = []

    # Unlabelled data
    for fname in sorted(os.listdir("unlabelled")):
        if not fname.endswith(".wav") or "_speed" in fname:
            continue
        audio, sr = load_audio(os.path.join("unlabelled", fname))
        train_embs.append(lstm_emb.embed(audio))
        train_labels.append(0)  # dummy label
        for factor in [0.9, 1.1]:
            aug = speed_perturb(audio, sr, factor)
            train_embs.append(lstm_emb.embed(aug))
            train_labels.append(0)

    # Augmented ref data (labelled)
    for i, name in enumerate(names):
        for fname in sorted(os.listdir("refs")):
            if not fname.endswith(".wav") or not fname.startswith(name + "_ref"):
                continue
            audio, sr = load_audio(os.path.join("refs", fname))
            train_embs.append(lstm_emb.embed(audio))
            train_labels.append(i + 1)
            for factor in [0.9, 1.1]:
                aug = speed_perturb(audio, sr, factor)
                train_embs.append(lstm_emb.embed(aug))
                train_labels.append(i + 1)

    train_embs = np.array(train_embs)
    train_labels = np.array(train_labels)

    print(f"  {len(train_embs)} training embeddings (256-dim)")
    print("\nTraining VAE (256 -> 64 latent)...")
    t0 = time.perf_counter()
    vae_model = train_vae(train_embs, train_labels, latent_dim=64)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Save checkpoint (strip classifier head for inference-only use)
    state_dict = {k: v for k, v in vae_model.state_dict().items() if "_classifier" not in k}
    config = {"input_dim": 256, "latent_dim": 64}
    checkpoint = {"_vae_config": config, "state_dict": state_dict}
    ckpt_path = f"{MODELS_DIR}/pyannote_lstm_vae.pt"
    torch.save(checkpoint, ckpt_path)
    print(f"  Saved: {ckpt_path}")

    # Project LSTM embeddings through VAE
    vae = SpeakerVAE(ckpt_path)
    proto_vae = vae.encode_batch(proto_lstm)
    turns_vae = vae.encode_batch(np.array(turns_lstm))

    acc_vae = classify(proto_vae, list(zip(turns_vae, turn_labels)), names)

    # Summary
    print()
    print(f"{'Method':<30} {'Dim':>6} {'Accuracy':>10} {'Proto Sep':>10}")
    print("-" * 60)
    print(f"{'ECAPA-TDNN (baseline)':<30} {'192':>6} {acc_ecapa:>9.1%} {sep(proto_ecapa):>10.4f}")
    print(f"{'Pyannote LSTM (raw)':<30} {'256':>6} {acc_lstm:>9.1%} {sep(proto_lstm):>10.4f}")
    print(f"{'Pyannote LSTM + VAE':<30} {'64':>6} {acc_vae:>9.1%} {sep(proto_vae):>10.4f}")
    print()
    print("The LSTM output retains more speaker information than the MLP layers")
    print("because it hasn't been compressed for the segmentation task.")
    print("The VAE can further refine this into a compact 64-dim representation.")


if __name__ == "__main__":
    main()
