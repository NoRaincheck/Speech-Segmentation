"""Few-shot speaker identification with VAE-enhanced embeddings.

Trains a small VAE on the reference embeddings, then uses the learned
latent space for improved speaker matching on individual dialogue turns.

Requires the VAE dependency group:
    uv run --group vae python examples/speaker_identification_vae.py

Example output::

    Loading embedding model...

    === Step 1: Generating few-shot reference samples ===
      Reusing 3 existing reference files for Bella
      Reusing 3 existing reference files for Bruno
      Reusing 3 existing reference files for Luna

    === Step 2: Building few-shot reference embeddings (ECAPA-TDNN) ===
        Bella bella_ref_0.wav: emb norm=1.0000
        Bella bella_ref_1.wav: emb norm=1.0000
        Bella bella_ref_2.wav: emb norm=1.0000
      Bella: averaged 3 embeddings -> prototype norm=1.0000
        Bruno bruno_ref_0.wav: emb norm=1.0000
        Bruno bruno_ref_1.wav: emb norm=1.0000
        Bruno bruno_ref_2.wav: emb norm=1.0000
      Bruno: averaged 3 embeddings -> prototype norm=1.0000
        Luna luna_ref_0.wav: emb norm=1.0000
        Luna luna_ref_1.wav: emb norm=1.0000
        Luna luna_ref_2.wav: emb norm=1.0000
      Luna: averaged 3 embeddings -> prototype norm=1.0000

    === Step 3: Training VAE on reference embeddings ===
      Collected 9 embeddings for VAE training (3 speakers)
      Reusing existing VAE model models/speaker_vae.pt

    === Step 4: Loading trained VAE ===
      VAE: 192 -> 128 (latent)

    === Step 5: Generating dialogue turns ===
      Reusing turns/turn_00_bella.wav
      Reusing turns/turn_01_bruno.wav
      Reusing turns/turn_02_bella.wav
      Reusing turns/turn_03_bruno.wav
      Reusing turns/turn_04_bella.wav
      Reusing turns/turn_05_bruno.wav
      Reusing turns/turn_06_bella.wav
      Reusing turns/turn_07_bruno.wav

    === Step 6: Classifying each turn against reference speakers ===

      Ground Truth   Predicted    Sim  Similarities
      ------------  ----------  -----  ------------------------------
             Bella       Bella  0.852  [Bella=0.852 Bruno=-0.422 Luna=0.307]
             Bruno       Bruno  0.990  [Bella=-0.234 Bruno=0.990 Luna=-0.407]
             Bella       Bella  0.934  [Bella=0.934 Bruno=-0.105 Luna=-0.013]
             Bruno       Bruno  0.988  [Bella=-0.174 Bruno=0.988 Luna=-0.424]
             Bella       Bella  0.879  [Bella=0.879 Bruno=-0.035 Luna=-0.079]
             Bruno       Bruno  0.973  [Bella=-0.179 Bruno=0.973 Luna=-0.356]
             Bella       Bella  0.851  [Bella=0.851 Bruno=-0.071 Luna=0.075]
             Bruno       Bruno  0.896  [Bella=-0.124 Bruno=0.896 Luna=-0.298]

      Accuracy: 8/8 turns correct (100.0%)
"""

import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from kittentts import KittenTTS

from speech_segmentation import SpeakerEmbedder
from speech_segmentation.vae import SpeakerVAE

EMB_MODEL_PATH = "models/ecapa_tdnn.onnx"
NORM_MEAN_PATH = "models/ecapa_norm_mean.npy"
VAE_MODEL_PATH = "models/speaker_vae.pt"
TTS_SR = 24000

VAE_INPUT_DIM = 192
VAE_LATENT_DIM = 128
VAE_NOISE_STD = 0.05
VAE_EPOCHS = 500
VAE_BATCH_SIZE = 32
VAE_LR = 1e-3
VAE_KL_WARMUP_EPOCHS = 100
VAE_CENTER_WEIGHT = 1.0
VAE_MARGIN_WEIGHT = 0.3
VAE_MARGIN = 1.0
VAE_HARD_NEG_WEIGHT = 0.1
VAE_HARD_NEG_MARGIN = 0.3
VAE_PROTO_AUG_SAMPLES = 3
VAE_PROTO_AUG_STD = 0.05

REFERENCE_PHRASES = {
    "Bella": [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "The early bird catches the worm but the second mouse gets the cheese.",
    ],
    "Bruno": [
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump.",
        "The five boxing wizards jump quickly across the mat.",
    ],
    "Luna": [
        "She sells seashells by the seashore every Sunday morning.",
        "Peter Piper picked a peck of pickled peppers.",
        "Unique New York, unique New York, you know you need unique New York.",
    ],
}

DIALOGUE = [
    ("Bella", "Hey Bruno, have you tried the new coffee shop on Main Street?"),
    ("Bruno", "Yes, I went there yesterday. The espresso was really good."),
    ("Bella", "I heard they also have great pastries. I want to try the croissants."),
    ("Bruno", "They do. The almond croissant is my favorite. You should definitely go."),
    ("Bella", "Sounds perfect. Want to go together this weekend?"),
    ("Bruno", "Sure, I was thinking Saturday morning. Shall we meet at ten?"),
    ("Bella", "Ten works great for me. I'll see you there then."),
    ("Bruno", "Looking forward to it. Have a great day, Bella."),
]


class _TrainingVAE(torch.nn.Module):
    """MLP-VAE used only during training (same architecture as SpeakerVAE)."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
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


def _compute_hard_neg_loss(mu_norm: torch.Tensor, labels: torch.Tensor, n_speakers: int) -> torch.Tensor:
    centers = torch.stack([mu_norm[labels == k].mean(0) for k in range(n_speakers)])
    all_sims = mu_norm @ centers.T
    mask = torch.eye(n_speakers, device=mu_norm.device)[labels].bool()
    all_sims = all_sims.masked_fill(mask, float("-inf"))
    hard_neg_sim = all_sims.max(dim=1).values
    return F.relu(hard_neg_sim - (1.0 - VAE_HARD_NEG_MARGIN)).mean()


def _sample_prototypes(
    mu_norm: torch.Tensor, labels: torch.Tensor, n_speakers: int
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = torch.stack([mu_norm[labels == k].mean(0) for k in range(n_speakers)])
    aug_embs = []
    aug_labels = []
    for k in range(n_speakers):
        noise = torch.randn(VAE_PROTO_AUG_SAMPLES, centers.shape[1], device=centers.device) * VAE_PROTO_AUG_STD
        samples = centers[k].unsqueeze(0) + noise
        samples = F.normalize(samples, dim=1)
        aug_embs.append(samples)
        aug_labels.append(torch.full((VAE_PROTO_AUG_SAMPLES,), k, dtype=torch.long, device=centers.device))
    return torch.cat(aug_embs), torch.cat(aug_labels)


def train_vae(embeddings: np.ndarray, labels: np.ndarray, save_path: str) -> None:
    if os.path.exists(save_path):
        print(f"  Reusing existing VAE model {save_path}")
        return
    n_speakers = len(set(labels))
    model = _TrainingVAE(VAE_INPUT_DIM, VAE_LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_LR, weight_decay=1e-4)

    data = torch.from_numpy(embeddings)
    lbl = torch.from_numpy(labels).long()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, lbl),
        batch_size=min(VAE_BATCH_SIZE, len(data)),
        shuffle=True,
    )

    print(f"  Training VAE: {VAE_INPUT_DIM} -> {VAE_LATENT_DIM} (latent), {VAE_EPOCHS} epochs")
    model.train()
    for epoch in range(1, VAE_EPOCHS + 1):
        beta = min(1.0, epoch / VAE_KL_WARMUP_EPOCHS)
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            noisy = batch_x + torch.randn_like(batch_x) * VAE_NOISE_STD
            recon, mu, logvar = model(noisy)
            recon_loss = F.mse_loss(recon, batch_x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            mu_norm = F.normalize(mu, dim=1)
            metric_mu = mu_norm * model._metric_scale
            metric_mu = F.normalize(metric_mu, dim=1)

            aug_embs, aug_labels = _sample_prototypes(metric_mu.detach(), batch_y, n_speakers)
            all_mu = torch.cat([metric_mu, aug_embs], dim=0)
            all_labels = torch.cat([batch_y, aug_labels], dim=0)

            centers = torch.stack([all_mu[all_labels == k].mean(0) for k in range(n_speakers)])
            center_loss = ((metric_mu - centers[batch_y]) ** 2).mean()

            margin_loss = torch.tensor(0.0, device=mu.device)
            for i in range(n_speakers):
                for j in range(i + 1, n_speakers):
                    dist = F.cosine_similarity(centers[i].unsqueeze(0), centers[j].unsqueeze(0))
                    margin_loss = margin_loss + F.relu(dist - (1.0 - VAE_MARGIN))

            hard_neg_loss = _compute_hard_neg_loss(metric_mu, batch_y, n_speakers)

            loss = (
                recon_loss
                + beta * kl_loss
                + VAE_CENTER_WEIGHT * center_loss
                + VAE_MARGIN_WEIGHT * margin_loss
                + VAE_HARD_NEG_WEIGHT * hard_neg_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 50 == 0 or epoch == 1:
            print(f"    Epoch {epoch:4d}/{VAE_EPOCHS}  loss={epoch_loss / len(loader):.6f}  beta={beta:.3f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {"_vae_config": {"input_dim": VAE_INPUT_DIM, "latent_dim": VAE_LATENT_DIM}, "state_dict": model.state_dict()},
        save_path,
    )
    print(f"  Saved VAE weights to {save_path}")


def generate_reference_samples(tts, ref_dir):
    os.makedirs(ref_dir, exist_ok=True)
    ref_paths = {}
    for voice, phrases in REFERENCE_PHRASES.items():
        paths = [os.path.join(ref_dir, f"{voice.lower()}_ref_{j}.wav") for j in range(len(phrases))]
        ref_paths[voice] = paths
        if all(os.path.exists(p) for p in paths):
            print(f"  Reusing {len(phrases)} existing reference files for {voice}")
            continue
        for j, phrase in enumerate(phrases):
            audio = tts.generate(phrase, voice=voice)
            sf.write(paths[j], audio, TTS_SR)
        print(f"  Generated {len(phrases)} reference files for {voice}")
    return ref_paths


def build_reference_embeddings(ref_paths, embedder):
    ref_embeddings = {}
    for voice, paths in ref_paths.items():
        voice_embs = []
        for path in paths:
            audio, _ = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            emb = embedder.embed(audio)
            voice_embs.append(emb)
            print(f"    {voice} {os.path.basename(path)}: emb norm={np.linalg.norm(emb):.4f}")
        avg_emb = np.mean(voice_embs, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        ref_embeddings[voice] = avg_emb
        print(f"  {voice}: averaged {len(voice_embs)} embeddings -> prototype norm={np.linalg.norm(avg_emb):.4f}")
    return ref_embeddings


def collect_all_embeddings(ref_paths, embedder):
    speaker_names = list(ref_paths.keys())
    embeddings = []
    labels = []
    for speaker_idx, (speaker, paths) in enumerate(ref_paths.items()):
        for path in paths:
            audio, _ = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            emb = embedder.embed(audio)
            embeddings.append(emb)
            labels.append(speaker_idx)
    arr = np.array(embeddings, dtype=np.float32)
    lbl = np.array(labels, dtype=np.int64)
    print(f"  Collected {len(arr)} embeddings for VAE training ({len(speaker_names)} speakers)")
    return arr, lbl


def generate_dialogue_turns(tts, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    turn_paths = []
    for i, (voice, text) in enumerate(DIALOGUE):
        path = os.path.join(output_dir, f"turn_{i:02d}_{voice.lower()}.wav")
        if os.path.exists(path):
            print(f"  Reusing {path}")
            turn_paths.append(path)
            continue
        audio = tts.generate(text, voice=voice)
        sf.write(path, audio, TTS_SR)
        turn_paths.append(path)
        print(f"  Generated {path} ({len(audio) / TTS_SR:.2f}s)")
    return turn_paths


def classify_turns(turn_paths, embedder, ref_embeddings, vae=None):
    ref_names = list(ref_embeddings.keys())
    ref_matrix = np.array([ref_embeddings[n] for n in ref_names])
    if vae is not None:
        ref_matrix = vae.encode_batch(ref_matrix)

    print(f"\n  {'Ground Truth':>12s}  {'Predicted':>10s}  {'Sim':>5s}  Similarities")
    print(f"  {'-' * 12}  {'-' * 10}  {'-' * 5}  {'-' * 30}")

    correct = 0
    for i, path in enumerate(turn_paths):
        gt_voice = DIALOGUE[i][0]
        audio, _ = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        emb = embedder.embed(audio)
        if vae is not None:
            emb = vae.encode(emb)
        emb = emb / np.linalg.norm(emb)
        sims = ref_matrix @ emb
        best_idx = int(sims.argmax())
        pred = ref_names[best_idx]
        is_correct = pred == gt_voice
        if is_correct:
            correct += 1
        mark = "" if is_correct else " [WRONG]"
        sim_str = " ".join(f"{n}={s:.3f}" for n, s in zip(ref_names, sims))
        print(f"  {gt_voice:>12s}  {pred:>10s}  {sims[best_idx]:5.3f}  [{sim_str}]{mark}")

    print(f"\n  Accuracy: {correct}/{len(turn_paths)} turns correct ({correct / len(turn_paths) * 100:.1f}%)")
    return correct, len(turn_paths)


def main():
    tts = KittenTTS("KittenML/kitten-tts-nano-0.8")
    print("Loading embedding model...")
    embedder = SpeakerEmbedder(EMB_MODEL_PATH, NORM_MEAN_PATH)

    print("\n=== Step 1: Generating few-shot reference samples ===")
    ref_paths = generate_reference_samples(tts, "refs")

    print("\n=== Step 2: Building few-shot reference embeddings (ECAPA-TDNN) ===")
    ref_embeddings = build_reference_embeddings(ref_paths, embedder)

    print("\n=== Step 3: Training VAE on reference embeddings ===")
    all_embs, all_labels = collect_all_embeddings(ref_paths, embedder)
    train_vae(all_embs, all_labels, VAE_MODEL_PATH)

    print("\n=== Step 4: Loading trained VAE ===")
    vae = SpeakerVAE(VAE_MODEL_PATH)
    print(f"  VAE: {vae.INPUT_DIM} -> {vae.LATENT_DIM} (latent)")

    print("\n=== Step 5: Generating dialogue turns ===")
    turn_paths = generate_dialogue_turns(tts, "turns")

    print("\n=== Step 6: Classifying each turn against reference speakers ===")
    classify_turns(turn_paths, embedder, ref_embeddings, vae=vae)


if __name__ == "__main__":
    main()
