"""VAE ablation: when and where does VAE improve speaker matching?

Trains VAEs with different strategies and compares against the raw
ECAPA-TDNN baseline on three metrics:
  1. Speaker identification accuracy (few-shot classification)
  2. Diarization Error Rate (DER)
  3. Embedding separation (intra-speaker vs inter-speaker cosine similarity)

Strategies compared:
  - No VAE (baseline: raw ECAPA-TDNN)
  - Unsupervised VAE (reconstruction + KL only)
  - VAE + Contrastive fine-tuning
  - VAE + Prototypical fine-tuning

Usage:
    uv run --group vae python demos/vae_ablation.py
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch

from speech_segmentation import SpeakerEmbedder, SpeechSegmenter
from speech_segmentation.audio import load_audio
from speech_segmentation.augmentation import speed_perturb
from speech_segmentation.diarizer import Diarizer
from speech_segmentation.evaluation import compute_diarization_error_rate
from speech_segmentation.vae import VAE, prototypical_loss, supervised_contrastive_loss

MODELS_DIR = "models"
TARGET_SR = 16000


def load_unlabelled_embeddings(embedder: SpeakerEmbedder, unlabelled_dir: str) -> np.ndarray:
    """Load and embed all unlabelled audio files with speed augmentation."""
    all_embs = []
    for fname in sorted(os.listdir(unlabelled_dir)):
        if not fname.endswith(".wav"):
            continue
        audio, sr = load_audio(os.path.join(unlabelled_dir, fname))
        emb = embedder.embed(audio)
        all_embs.append(emb)
        for factor in [0.9, 1.1]:
            aug = speed_perturb(audio, sr, factor)
            all_embs.append(embedder.embed(aug))
    return np.array(all_embs)


def train_vae_unsupervised(
    embeddings: np.ndarray, latent_dim: int, free_bits: float, epochs: int = 500
) -> VAE:
    """Train VAE with reconstruction + KL loss only."""
    input_dim = embeddings.shape[1]
    model = VAE(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.from_numpy(embeddings).float()
    dataset = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    kl_weight = 0.0
    warmup_epochs = 100
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            kl_weight = min(1.0, epoch / warmup_epochs)
        else:
            kl_weight = 1.0

        total_loss = 0.0
        for (batch,) in loader:
            recon, mu, logvar = model(batch)
            recon_loss = torch.nn.functional.mse_loss(recon, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(batch)
            kl_loss = torch.clamp(kl_loss, min=free_bits)
            loss = recon_loss + kl_weight * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def train_vae_contrastive(
    embeddings: np.ndarray, labels: np.ndarray, latent_dim: int, free_bits: float, epochs: int = 500
) -> VAE:
    """Train VAE unsupervised first, then fine-tune with contrastive loss."""
    n_speakers = len(np.unique(labels))
    base_model = train_vae_unsupervised(embeddings, latent_dim, free_bits, epochs)

    model = VAE(embeddings.shape[1], latent_dim, num_speakers=n_speakers)
    model.load_state_dict(base_model.state_dict(), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(labels).long()
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(200):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            mu, logvar = model.encode(batch_x)
            z = mu
            z_norm = torch.nn.functional.normalize(z, dim=1)
            loss = supervised_contrastive_loss(z_norm, batch_y, temperature=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def train_vae_prototypical(
    embeddings: np.ndarray, labels: np.ndarray, latent_dim: int, free_bits: float, epochs: int = 500
) -> VAE:
    """Train VAE unsupervised first, then fine-tune with prototypical loss."""
    n_speakers = len(np.unique(labels))
    base_model = train_vae_unsupervised(embeddings, latent_dim, free_bits, epochs)

    model = VAE(embeddings.shape[1], latent_dim, num_speakers=n_speakers)
    model.load_state_dict(base_model.state_dict(), strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(labels).long()
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(200):
        for batch_x, batch_y in loader:
            mu, logvar = model.encode(batch_x)
            z = mu
            z_norm = torch.nn.functional.normalize(z, dim=1)
            loss = prototypical_loss(z_norm, batch_y, n_speakers, temperature=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def compute_embedding_separation(embeddings: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute mean intra-speaker and inter-speaker cosine similarity."""
    unique = np.unique(labels)
    intra_sims = []
    inter_sims = []

    for spk in unique:
        mask = labels == spk
        spk_embs = embeddings[mask]
        if len(spk_embs) < 2:
            continue
        for i in range(len(spk_embs)):
            for j in range(i + 1, len(spk_embs)):
                intra_sims.append(float(np.dot(spk_embs[i], spk_embs[j])))

    for i, spk_a in enumerate(unique):
        for spk_b in unique[i + 1 :]:
            embs_a = embeddings[labels == spk_a]
            embs_b = embeddings[labels == spk_b]
            for a in embs_a:
                for b in embs_b:
                    inter_sims.append(float(np.dot(a, b)))

    mean_intra = float(np.mean(intra_sims)) if intra_sims else 0.0
    mean_inter = float(np.mean(inter_sims)) if inter_sims else 0.0
    return mean_intra, mean_inter


def evaluate_few_shot(
    ref_embeddings: dict[str, np.ndarray],
    turn_embeddings: dict[str, list[np.ndarray]],
    names: list[str],
) -> float:
    """Evaluate few-shot classification accuracy using prototype matching."""
    proto_matrix = np.array([ref_embeddings[n] for n in names])
    norms = np.linalg.norm(proto_matrix, axis=1, keepdims=True)
    proto_matrix = proto_matrix / norms

    correct = 0
    total = 0
    for name, embs in turn_embeddings.items():
        for emb in embs:
            sims = proto_matrix @ emb
            pred = names[int(np.argmax(sims))]
            correct += int(pred == name)
            total += 1
    return correct / total if total else 0.0


def evaluate_diarization(
    diarizer: Diarizer,
    turn_dir: str,
    ref_embeddings: dict[str, np.ndarray],
    stitch_threshold: float | None = None,
) -> float:
    """Build a combined audio from turns and evaluate DER."""
    turn_files = sorted(f for f in os.listdir(turn_dir) if f.endswith(".wav"))
    parts = []
    for fname in turn_files:
        audio, _ = load_audio(os.path.join(turn_dir, fname))
        parts.append(audio)
        parts.append(np.zeros(int(0.3 * TARGET_SR), dtype=np.float32))

    combined = np.concatenate(parts)
    diarizer.build_references(ref_embeddings)
    matches, _ = diarizer.diarize(combined, stitch_threshold=stitch_threshold)

    ground_truth = []
    offset = 0.0
    for fname in turn_files:
        speaker = fname.split("_")[2].replace(".wav", "")
        audio, _ = load_audio(os.path.join(turn_dir, fname))
        duration = len(audio) / TARGET_SR
        ground_truth.append((offset, offset + duration, speaker))
        offset += duration + 0.3

    hypothesis = [(m.start_time, m.end_time, m.speaker) for m in matches]
    metrics = compute_diarization_error_rate(ground_truth, hypothesis)
    return metrics.der


def main():
    parser = argparse.ArgumentParser(description="VAE ablation study")
    parser.add_argument("--refs-dir", default="refs")
    parser.add_argument("--turns-dir", default="turns")
    parser.add_argument("--unlabelled-dir", default="unlabelled")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--free-bits", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    print("Loading models...")
    embedder = SpeakerEmbedder(f"{MODELS_DIR}/ecapa_tdnn.onnx")
    segmenter = SpeechSegmenter(f"{MODELS_DIR}/model.onnx")

    # Build reference embeddings
    print("Building reference embeddings...")
    speaker_files: dict[str, list[str]] = defaultdict(list)
    for fname in sorted(os.listdir(args.refs_dir)):
        if not fname.endswith(".wav"):
            continue
        speaker = fname.rsplit("_ref_", 1)[0]
        speaker_files[speaker].append(os.path.join(args.refs_dir, fname))

    names = sorted(speaker_files.keys())
    ref_embeddings = {}
    for name in names:
        embs = [embedder.embed(load_audio(p)[0]) for p in speaker_files[name]]
        ref_embeddings[name] = np.mean(embs, axis=0)
        ref_embeddings[name] = ref_embeddings[name] / np.linalg.norm(ref_embeddings[name])

    # Build turn embeddings
    turn_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    for fname in sorted(os.listdir(args.turns_dir)):
        if not fname.endswith(".wav"):
            continue
        speaker = fname.split("_")[2].replace(".wav", "")
        audio, _ = load_audio(os.path.join(args.turns_dir, fname))
        turn_embeddings[speaker].append(embedder.embed(audio))

    # Load unlabelled data for VAE training
    print("Loading unlabelled embeddings for VAE training...")
    t0 = time.perf_counter()
    unlabelled_embs = load_unlabelled_embeddings(embedder, args.unlabelled_dir)
    print(f"  {len(unlabelled_embs)} embeddings in {time.perf_counter() - t0:.1f}s")

    # Generate synthetic labels from reference data for contrastive/prototypical training
    train_embs = []
    train_labels = []
    for i, name in enumerate(names):
        for p in speaker_files[name]:
            audio, _ = load_audio(p)
            train_embs.append(embedder.embed(audio))
            train_labels.append(i)
    train_embs = np.array(train_embs)
    train_labels = np.array(train_labels)

    # Also add augmented reference embeddings
    aug_embs = []
    aug_labels = []
    for i, name in enumerate(names):
        for p in speaker_files[name]:
            audio, sr = load_audio(p)
            for factor in [0.9, 1.1]:
                aug = speed_perturb(audio, sr, factor)
                aug_embs.append(embedder.embed(aug))
                aug_labels.append(i)
    train_embs = np.concatenate([train_embs, np.array(aug_embs)])
    train_labels = np.concatenate([train_labels, np.array(aug_labels)])

    strategies = {
        "No VAE (baseline)": None,
        "Unsupervised VAE": lambda: train_vae_unsupervised(
            unlabelled_embs, args.latent_dim, args.free_bits, args.epochs
        ),
        "VAE + Contrastive FT": lambda: train_vae_contrastive(
            train_embs, train_labels, args.latent_dim, args.free_bits, args.epochs
        ),
        "VAE + Prototypical FT": lambda: train_vae_prototypical(
            train_embs, train_labels, args.latent_dim, args.free_bits, args.epochs
        ),
    }

    results = []
    for name, train_fn in strategies.items():
        print(f"\n{'=' * 60}")
        print(f"Strategy: {name}")
        print(f"{'=' * 60}")

        t0 = time.perf_counter()

        if train_fn is not None:
            vae_model = train_fn()
            # Save and reload through SpeakerVAE to test inference path
            config = {
                "input_dim": unlabelled_embs.shape[1],
                "latent_dim": args.latent_dim,
            }
            if vae_model._classifier is not None:
                config["num_speakers"] = vae_model._classifier.out_features
            checkpoint = {
                "_vae_config": config,
                "state_dict": vae_model.state_dict(),
            }
            tmp_path = f"/tmp/vae_{name.replace(' ', '_').replace('+', '').lower()}.pt"
            torch.save(checkpoint, tmp_path)

            from speech_segmentation.vae import SpeakerVAE

            vae = SpeakerVAE(tmp_path)
            os.remove(tmp_path)

            # Project all embeddings through VAE
            projected_refs = {
                n: vae.encode(ref_embeddings[n]) for n in names
            }
            projected_turns = {
                n: [vae.encode(e) for e in embs] for n, embs in turn_embeddings.items()
            }
            all_projected = vae.encode_batch(
                np.concatenate([unlabelled_embs, train_embs])
            )
            unlabelled_projected = all_projected[: len(unlabelled_embs)]
            train_projected = all_projected[len(unlabelled_embs) :]

            diarizer = Diarizer(segmenter, embedder, vae=vae)
        else:
            vae = None
            projected_refs = ref_embeddings
            projected_turns = turn_embeddings
            unlabelled_projected = unlabelled_embs
            train_projected = train_embs
            diarizer = Diarizer(segmenter, embedder)

        train_time = time.perf_counter() - t0

        # Metric 1: Speaker ID accuracy
        acc = evaluate_few_shot(projected_refs, projected_turns, names)

        # Metric 2: DER
        der = evaluate_diarization(diarizer, args.turns_dir, ref_embeddings, stitch_threshold=0.7)

        # Metric 3: Embedding separation
        all_embs_for_sep = np.concatenate([unlabelled_projected, train_projected])
        all_labels_for_sep = np.concatenate([
            np.full(len(unlabelled_projected), -1),
            train_labels,
        ])
        # Only use labelled data for separation
        intra, inter = compute_embedding_separation(train_projected, train_labels)
        separation_gap = intra - inter

        print(f"  Training time: {train_time:.1f}s")
        print(f"  Speaker ID accuracy: {acc:.1%}")
        print(f"  DER: {der:.4f} ({der * 100:.2f}%)")
        print(f"  Embedding separation: intra={intra:.4f}, inter={inter:.4f}, gap={separation_gap:.4f}")

        results.append({
            "strategy": name,
            "accuracy": acc,
            "der": der,
            "intra_sim": intra,
            "inter_sim": inter,
            "separation_gap": separation_gap,
            "train_time": train_time,
        })

    # Summary table
    print(f"\n{'=' * 80}")
    print("ABLACTION SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Strategy':<25} {'Acc':>8} {'DER':>8} {'Intra':>8} {'Inter':>8} {'Gap':>8} {'Time':>8}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['strategy']:<25} {r['accuracy']:>7.1%} {r['der']:>7.4f} "
            f"{r['intra_sim']:>7.4f} {r['inter_sim']:>7.4f} {r['separation_gap']:>7.4f} "
            f"{r['train_time']:>6.1f}s"
        )

    # Delta analysis
    baseline = results[0]
    print(f"\nDelta vs baseline ({baseline['strategy']}):")
    for r in results[1:]:
        acc_delta = r["accuracy"] - baseline["accuracy"]
        der_delta = r["der"] - baseline["der"]
        gap_delta = r["separation_gap"] - baseline["separation_gap"]
        print(
            f"  {r['strategy']:<25} acc={acc_delta:+.1%}  DER={der_delta:+.4f}  "
            f"sep_gap={gap_delta:+.4f}"
        )


if __name__ == "__main__":
    main()
