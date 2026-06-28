"""Few-shot speech diarization with VAE-enhanced embeddings.

Trains a small VAE on the reference embeddings, then uses the learned
latent space for improved few-shot speaker matching.

Requires the VAE dependency group:
    uv run --group vae python examples/few_shot_diarization_vae.py
"""

import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from kittentts import KittenTTS

from speech_segmentation import Diarizer, SpeakerEmbedder, SpeechSegmenter
from speech_segmentation.vae import SpeakerVAE

SEG_MODEL_PATH = "models/model.onnx"
EMB_MODEL_PATH = "models/ecapa_tdnn.onnx"
NORM_MEAN_PATH = "models/ecapa_norm_mean.npy"
VAE_MODEL_PATH = "models/speaker_vae.pt"
TTS_SR = 24000
SILENCE_DURATION = 0.5

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
        # Learned diagonal metric: amplifies discriminative latent dimensions.
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
    """For each sample, penalize the cosine similarity to its closest different-speaker center."""
    centers = torch.stack([mu_norm[labels == k].mean(0) for k in range(n_speakers)])
    # cosine similarity of each sample to all centers: (batch, n_speakers)
    all_sims = mu_norm @ centers.T
    # mask out the same-speaker center (set to -inf so it won't be selected)
    mask = torch.eye(n_speakers, device=mu_norm.device)[labels].bool()
    all_sims = all_sims.masked_fill(mask, float("-inf"))
    # hardest negative = highest similarity to a different-speaker center
    hard_neg_sim = all_sims.max(dim=1).values
    # penalize if the hardest negative is too similar (above 1 - margin)
    return F.relu(hard_neg_sim - (1.0 - VAE_HARD_NEG_MARGIN)).mean()


def _sample_prototypes(
    mu_norm: torch.Tensor, labels: torch.Tensor, n_speakers: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample synthetic embeddings from Gaussian around each class center.

    Returns concatenated (original + synthetic) embeddings and labels.
    """
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
    """Train VAE on ECAPA-TDNN embeddings with center loss, hard negative mining, and prototype augmentation."""
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
    print(f"  Losses: recon + KL + {VAE_CENTER_WEIGHT:.2f}*center + {VAE_MARGIN_WEIGHT:.2f}*margin")
    print(
        f"          + {VAE_HARD_NEG_WEIGHT:.2f}*hard_neg({VAE_HARD_NEG_MARGIN}) + proto_aug({VAE_PROTO_AUG_SAMPLES}x{VAE_PROTO_AUG_STD})"
    )
    model.train()
    for epoch in range(1, VAE_EPOCHS + 1):
        beta = min(1.0, epoch / VAE_KL_WARMUP_EPOCHS)
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            # --- VAE forward pass ---
            noisy = batch_x + torch.randn_like(batch_x) * VAE_NOISE_STD
            recon, mu, logvar = model(noisy)
            recon_loss = F.mse_loss(recon, batch_x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            mu_norm = F.normalize(mu, dim=1)
            metric_mu = mu_norm * model._metric_scale
            metric_mu = F.normalize(metric_mu, dim=1)

            # --- Prototype augmentation: synthetic latent points near each center ---
            # These are already in metric space, used only for center/margin losses.
            aug_embs, aug_labels = _sample_prototypes(metric_mu.detach(), batch_y, n_speakers)

            # Combine real + augmented for center/hard_neg losses
            all_mu = torch.cat([metric_mu, aug_embs], dim=0)
            all_labels = torch.cat([batch_y, aug_labels], dim=0)

            # --- Center loss on combined set ---
            centers = torch.stack([all_mu[all_labels == k].mean(0) for k in range(n_speakers)])
            center_loss = ((metric_mu - centers[batch_y]) ** 2).mean()

            # --- Margin loss: push different-speaker centers apart ---
            margin_loss = torch.tensor(0.0, device=mu.device)
            for i in range(n_speakers):
                for j in range(i + 1, n_speakers):
                    dist = F.cosine_similarity(centers[i].unsqueeze(0), centers[j].unsqueeze(0))
                    margin_loss = margin_loss + F.relu(dist - (1.0 - VAE_MARGIN))

            # --- Hard negative mining on real embeddings ---
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
            if len(audio) != 16000 * len(audio) // 16000:
                num_samples = int(len(audio) * 16000 / 24000)
                audio = np.interp(np.linspace(0, len(audio) - 1, num_samples), np.arange(len(audio)), audio)
            emb = embedder.embed(audio)
            voice_embs.append(emb)
            print(f"    {voice} {os.path.basename(path)}: emb norm={np.linalg.norm(emb):.4f}")
        avg_emb = np.mean(voice_embs, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        ref_embeddings[voice] = avg_emb
        print(f"  {voice}: averaged {len(voice_embs)} embeddings -> prototype norm={np.linalg.norm(avg_emb):.4f}")
    return ref_embeddings


def collect_all_embeddings(ref_paths, embedder):
    """Extract raw ECAPA-TDNN embeddings from all reference files."""
    speaker_names = list(ref_paths.keys())
    embeddings = []
    labels = []
    for speaker_idx, (speaker, paths) in enumerate(ref_paths.items()):
        for path in paths:
            audio, _ = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if len(audio) != 16000 * len(audio) // 16000:
                num_samples = int(len(audio) * 16000 / 24000)
                audio = np.interp(np.linspace(0, len(audio) - 1, num_samples), np.arange(len(audio)), audio)
            emb = embedder.embed(audio)
            embeddings.append(emb)
            labels.append(speaker_idx)
    arr = np.array(embeddings, dtype=np.float32)
    lbl = np.array(labels, dtype=np.int64)
    print(f"  Collected {len(arr)} embeddings for VAE training ({len(speaker_names)} speakers)")
    return arr, lbl


def generate_conversation(tts, output_path):
    boundaries_path = output_path.rsplit(".", 1)[0] + "_turns.npy"
    if os.path.exists(output_path) and os.path.exists(boundaries_path):
        info = sf.info(output_path)
        print(f"  Reusing existing {output_path} ({info.duration:.2f}s)")
        return sf.read(output_path, dtype="float32")[0], np.load(boundaries_path, allow_pickle=True)
    silence = np.zeros(int(TTS_SR * SILENCE_DURATION), dtype=np.float32)
    parts = []
    turn_durations = []
    for i, (voice, text) in enumerate(DIALOGUE):
        audio = tts.generate(text, voice=voice)
        parts.append(audio)
        turn_durations.append(len(audio) / TTS_SR)
        if i < len(DIALOGUE) - 1:
            parts.append(silence)
    conversation = np.concatenate(parts)
    sf.write(output_path, conversation, TTS_SR)
    turn_boundaries = []
    cum_time = 0.0
    for idx, ((voice, _), dur) in enumerate(zip(DIALOGUE, turn_durations)):
        turn_boundaries.append((cum_time, cum_time + dur, voice))
        cum_time += dur + (SILENCE_DURATION if idx < len(DIALOGUE) - 1 else 0)
    np.save(boundaries_path, np.array(turn_boundaries, dtype=object))
    print(f"  Generated {output_path} ({len(conversation) / TTS_SR:.2f}s, {len(DIALOGUE)} turns)")
    return conversation, turn_boundaries


def extract_per_speaker_audio(matches, conv_audio_24k):
    speaker_segments = {}
    for m in matches:
        name = m.speaker
        speaker_segments.setdefault(name, []).append(m)

    results = {}
    for name, segs in speaker_segments.items():
        parts = []
        for m in segs:
            start_16k = m.start_frame * 270
            end_16k = m.end_frame * 270
            start_24k = int(start_16k * TTS_SR / 16000)
            end_24k = int(end_16k * TTS_SR / 16000)
            start_24k = min(start_24k, len(conv_audio_24k))
            end_24k = min(end_24k, len(conv_audio_24k))
            if start_24k < end_24k:
                parts.append(conv_audio_24k[start_24k:end_24k])
        if parts:
            audio = np.concatenate(parts)
            path = f"{name.lower()}.wav"
            sf.write(path, audio, TTS_SR)
            results[name] = {"path": path, "duration": len(audio) / TTS_SR, "segments": len(segs)}
    return results


def main():
    tts = KittenTTS("KittenML/kitten-tts-nano-0.8")
    print("Loading ONNX models...")
    segmenter = SpeechSegmenter(SEG_MODEL_PATH)
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

    diarizer = Diarizer(segmenter, embedder, vae=vae)
    diarizer.build_references(ref_embeddings)

    print("\n=== Step 5: Generating conversation ===")
    conv_audio, turn_boundaries = generate_conversation(tts, "conversation.wav")

    print("\n=== Step 6: Segmenting conversation (pyannote) ===")
    conv_audio_16k, _ = sf.read("conversation.wav", dtype="float32")
    if conv_audio_16k.ndim > 1:
        conv_audio_16k = conv_audio_16k.mean(axis=1)

    print("\n=== Step 7: Few-shot speaker matching (VAE latent) ===")
    matches, segments = diarizer.diarize(conv_audio_16k)
    print(f"  Detected {len(segments)} speech segments")
    for i, m in enumerate(matches):
        sim_str = " ".join(f"{k}={v:.3f}" for k, v in m.all_sims.items())
        print(
            f"  Seg {i:2d} ({m.start_time:5.2f}s-{m.end_time:5.2f}s): "
            f"{m.speaker:5s} (sim={m.similarity:.3f}) [{sim_str}]"
        )

    print("\n  --- Reference Similarity Summary ---")
    for name in REFERENCE_PHRASES:
        sims = [m.all_sims[name] for m in matches]
        avg = np.mean(sims)
        is_control = name == "Luna"
        marker = " (control - should be low)" if is_control else ""
        print(f"  {name:6s} avg sim: {avg:.3f}{marker}")

    print("\n=== Step 8: Extracting per-speaker audio ===")
    conv_audio_24k, _ = sf.read("conversation.wav", dtype="float32")
    results = extract_per_speaker_audio(matches, conv_audio_24k)
    for name, info in sorted(results.items()):
        print(f"  {name}: {info['segments']} segments, {info['duration']:.1f}s total -> {info['path']}")

    print("\n=== Summary ===")

    def _ground_truth(mid_time: float) -> str:
        for t_start, t_end, voice in turn_boundaries:
            if t_start <= mid_time < t_end:
                return voice
        return "Unknown"

    for i, m in enumerate(matches):
        mid = (m.start_time + m.end_time) / 2
        gt_voice = _ground_truth(mid)
        correct = m.speaker == gt_voice
        mark = "" if correct else " [WRONG]"
        print(f"  Seg {i:2d} ({m.start_time:5.2f}s-{m.end_time:5.2f}s): {m.speaker:5s} sim={m.similarity:.3f}  gt={gt_voice:5s}{mark}")

    correct_count = sum(
        1 for m in matches
        if _ground_truth((m.start_time + m.end_time) / 2) != "Unknown"
        and m.speaker == _ground_truth((m.start_time + m.end_time) / 2)
    )
    total = sum(
        1 for m in matches
        if _ground_truth((m.start_time + m.end_time) / 2) != "Unknown"
    )
    print(f"\n  Accuracy: {correct_count}/{total} segments correct ({correct_count / total * 100:.1f}%)")

    for name in REFERENCE_PHRASES:
        segs_for_name = [m for m in matches if m.speaker == name]
        if segs_for_name:
            avg_sim = np.mean([m.similarity for m in segs_for_name])
            label = "CONTROL (should be absent)" if name == "Luna" else "SPEAKER"
            print(f"  {name:6s}: {len(segs_for_name):2d} segments, avg sim={avg_sim:.3f} [{label}]")
        else:
            label = "correctly absent" if name == "Luna" else "not detected"
            print(f"  {name:6s}: 0 segments [{label}]")


if __name__ == "__main__":
    main()
