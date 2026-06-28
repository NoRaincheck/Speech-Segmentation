"""Few-shot speaker identification with VAE trained on unlabelled samples.

Trains a small VAE on unlabelled audio embeddings, then uses the learned
latent space for improved speaker matching on individual dialogue turns.

The VAE learns a general speech embedding space from unlabelled data,
while the few-shot reference embeddings provide speaker-specific prototypes.

Requires the VAE dependency group:
    uv run --group vae python examples/unsupervised_vae_speaker_id.py
"""

import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from kittentts import KittenTTS

from speech_segmentation import SpeakerEmbedder
from speech_segmentation.vae import SpeakerVAE, VAE

EMB_MODEL_PATH = "models/ecapa_tdnn.onnx"
NORM_MEAN_PATH = "models/ecapa_norm_mean.npy"
VAE_MODEL_PATH = "models/unsupervised_vae.pt"
TTS_SR = 24000

VAE_INPUT_DIM = 192
VAE_LATENT_DIM = 128
VAE_NOISE_STD = 0.05
VAE_EPOCHS = 500
VAE_BATCH_SIZE = 32
VAE_LR = 1e-3
VAE_KL_WARMUP_EPOCHS = 100

UNLABELLED_PHRASES = [
    "The sun rises in the east and sets in the west.",
    "Practice makes perfect when you dedicate yourself.",
    "Knowledge is power but enthusiasm pulls the switch.",
    "The best time to plant a tree was twenty years ago.",
    "Success is not final and failure is not fatal.",
    "In the middle of difficulty lies opportunity.",
    "Life is what happens when you are busy making other plans.",
    "The only way to do great work is to love what you do.",
    "Stay hungry stay foolish and never stop learning.",
    "Innovation distinguishes between a leader and a follower.",
]

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


def train_vae(embeddings: np.ndarray, save_path: str) -> None:
    if os.path.exists(save_path):
        print(f"  Reusing existing VAE model {save_path}")
        return
    model = VAE(VAE_INPUT_DIM, VAE_LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_LR, weight_decay=1e-4)

    data = torch.from_numpy(embeddings)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=min(VAE_BATCH_SIZE, len(data)),
        shuffle=True,
    )

    print(f"  Training VAE: {VAE_INPUT_DIM} -> {VAE_LATENT_DIM} (latent), {VAE_EPOCHS} epochs")
    model.train()
    for epoch in range(1, VAE_EPOCHS + 1):
        beta = min(1.0, epoch / VAE_KL_WARMUP_EPOCHS)
        epoch_loss = 0.0
        for (batch_x,) in loader:
            noisy = batch_x + torch.randn_like(batch_x) * VAE_NOISE_STD
            recon, mu, logvar = model(noisy)
            recon_loss = F.mse_loss(recon, batch_x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss
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


def build_reference_embeddings(ref_dir, embedder):
    voices = ["Bella", "Bruno", "Luna"]
    ref_paths = {v: [os.path.join(ref_dir, f"{v.lower()}_ref_{j}.wav") for j in range(3)] for v in voices}
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


UNLABELLED_VOICES = ["Bella", "Bruno", "Luna"]


def generate_unlabelled_samples(tts, unlabelled_dir):
    os.makedirs(unlabelled_dir, exist_ok=True)
    unlabelled_paths = []
    for i, phrase in enumerate(UNLABELLED_PHRASES):
        path = os.path.join(unlabelled_dir, f"unlabelled_{i:02d}.wav")
        if os.path.exists(path):
            print(f"  Reusing {path}")
            unlabelled_paths.append(path)
            continue
        voice = UNLABELLED_VOICES[i % len(UNLABELLED_VOICES)]
        audio = tts.generate(phrase, voice=voice)
        sf.write(path, audio, TTS_SR)
        unlabelled_paths.append(path)
        print(f"  Generated {path} ({len(audio) / TTS_SR:.2f}s)")
    return unlabelled_paths


def collect_unlabelled_embeddings(unlabelled_paths, embedder):
    embeddings = []
    for path in unlabelled_paths:
        audio, _ = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        emb = embedder.embed(audio)
        embeddings.append(emb)
    arr = np.array(embeddings, dtype=np.float32)
    print(f"  Collected {len(arr)} unlabelled embeddings for VAE training")
    return arr


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

    print("\n=== Step 1: Building few-shot reference embeddings (ECAPA-TDNN) ===")
    ref_embeddings = build_reference_embeddings("refs", embedder)

    print("\n=== Step 2: Generating unlabelled samples for VAE training ===")
    unlabelled_paths = generate_unlabelled_samples(tts, "unlabelled")

    print("\n=== Step 3: Collecting unlabelled embeddings ===")
    unlabelled_embs = collect_unlabelled_embeddings(unlabelled_paths, embedder)

    print("\n=== Step 4: Training VAE on unlabelled embeddings ===")
    train_vae(unlabelled_embs, VAE_MODEL_PATH)

    print("\n=== Step 5: Loading trained VAE ===")
    vae = SpeakerVAE(VAE_MODEL_PATH)
    print(f"  VAE: {vae.INPUT_DIM} -> {vae.LATENT_DIM} (latent)")

    print("\n=== Step 6: Generating dialogue turns ===")
    turn_paths = generate_dialogue_turns(tts, "turns")

    print("\n=== Step 7: Classifying each turn against reference speakers ===")
    classify_turns(turn_paths, embedder, ref_embeddings, vae=vae)


if __name__ == "__main__":
    main()
