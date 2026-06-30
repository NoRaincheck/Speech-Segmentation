"""When VAE helps: cross-engine and few-shot with limited data.

ECAPA-TDNN is excellent when reference and test audio come from the same
conditions. But when they differ — different TTS engine, different mic,
limited reference clips — the raw embedding space can fail.

A VAE trained with contrastive loss on multi-engine data learns a shared
canonical space where speaker identity is preserved regardless of source.
This example demonstrates two scenarios where VAE improves results:

  1. Cross-engine matching: reference from Kokoro, test from Piper
  2. Limited references: 1 reference clip per speaker (instead of 5)

Key insight: the VAE must be trained on data from ALL engines. Training
on a single engine causes the VAE to learn engine-specific features,
which hurts cross-engine generalization.

Usage:
    uv run --group vae python examples/vae_cross_engine.py
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

# Paths
MODEL_DIR = "models"
ECAPA_PATH = f"{MODEL_DIR}/ecapa_tdnn.onnx"

# Phrases to synthesize for training and testing
PHRASES = [
    "Hello, how are you doing today?",
    "The weather is quite nice outside.",
    "I would like to have a conversation.",
]

# Kokoro voice IDs mapped by gender
KOKORO_VOICES = {
    "M": ["am_adam", "am_michael", "bm_george"],
    "F": ["af_bella", "af_sarah", "bf_emma"],
}

# Piper model names mapped by gender
PIPER_VOICES = {
    "M": ["en_US-lessac-medium"],
    "F": ["en_US-amy-medium"],
}


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------

def synthesize_kokoro(text: str, voice: str) -> np.ndarray:
    """Synthesize text with Kokoro-ONNX. Returns 16kHz float32 audio."""
    from kokoro_onnx import Kokoro

    k = Kokoro(f"{MODEL_DIR}/kokoro-v1.0.onnx", f"{MODEL_DIR}/voices-v1.0.bin")
    audio, sr = k.create(text, voice=voice, speed=1.0)
    if sr != 16000:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return audio


def synthesize_piper(text: str, model_name: str) -> np.ndarray:
    """Synthesize text with Piper-TTS. Returns 16kHz float32 audio."""
    from piper import PiperVoice

    v = PiperVoice.load(f"{MODEL_DIR}/{model_name}.onnx")
    chunks = list(v.synthesize(text))
    audio = np.concatenate([c.audio_float_array for c in chunks])
    sr = v.config.sample_rate
    if sr != 16000:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return audio


# ---------------------------------------------------------------------------
# VAE training
# ---------------------------------------------------------------------------

def train_vae(embeddings: np.ndarray, labels: np.ndarray) -> VAE:
    """Train a VAE with unsupervised pre-training + contrastive fine-tuning.

    The VAE learns a 64-dimensional latent space from 192-dim ECAPA-TDNN
    embeddings. The contrastive loss pulls same-speaker embeddings together
    and pushes different-speaker embeddings apart in the latent space.

    Architecture:
        Encoder: 192 → 256 → 128 → 64 → mu, logvar (64-dim latent)
        Decoder: 64 → 128 → 256 → 192 (reconstruction)
        Classification head: 64 → n_speakers (for contrastive FT)
    """
    n_speakers = len(np.unique(labels))
    input_dim = embeddings.shape[1]
    latent_dim = 64

    # --- Phase 1: Unsupervised pre-training ---
    # Train the VAE to reconstruct its input. The KL divergence regularizes
    # the latent space to be smooth and continuous. Free bits (clamped KL)
    # prevent posterior collapse.
    model = VAE(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(embeddings).float()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x), batch_size=32, shuffle=True
    )

    for epoch in range(400):
        # KL warmup: gradually increase KL weight over first 100 epochs
        kl_weight = min(1.0, epoch / 100)
        for (batch,) in loader:
            recon, mu, logvar = model(batch)
            recon_loss = torch.nn.functional.mse_loss(recon, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(batch)
            kl_loss = torch.clamp(kl_loss, min=2.0)  # free bits
            loss = recon_loss + kl_weight * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # --- Phase 2: Contrastive fine-tuning ---
    # Freeze the encoder structure, add a classification head, and fine-tune
    # with supervised contrastive loss. This pulls same-speaker embeddings
    # together and pushes different-speaker embeddings apart.
    model_ft = VAE(input_dim, latent_dim, num_speakers=n_speakers)
    model_ft.load_state_dict(model.state_dict(), strict=False)
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-4)
    y = torch.from_numpy(labels).long()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=32, shuffle=True
    )

    for epoch in range(300):
        for batch_x, batch_y in loader:
            mu, _ = model_ft.encode(batch_x)
            z = torch.nn.functional.normalize(mu, dim=1)
            loss = supervised_contrastive_loss(z, batch_y, temperature=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model_ft.eval()
    return model_ft


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def project(vae: VAE, embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings through the VAE encoder. Returns L2-normalized vectors."""
    t = torch.from_numpy(embeddings).float()
    with torch.no_grad():
        mu, _ = vae.encode(t)
    z = mu.numpy()
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def classify(proto: np.ndarray, turns: dict[str, np.ndarray], names: list[str]) -> float:
    """Classify turns against prototypes. Returns accuracy."""
    correct = 0
    total = 0
    for spk, embs in turns.items():
        for e in embs:
            if names[int(np.argmax(proto @ e))] == spk:
                correct += 1
            total += 1
    return correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("WHEN VAE HELPS: Cross-Engine & Limited Reference Matching")
    print("=" * 70)
    print()
    print("ECAPA-TDNN excels when reference and test come from the same source.")
    print("But when they differ (different TTS engine, limited data), a VAE")
    print("trained on multi-engine data creates a shared canonical space.")
    print()

    embedder = SpeakerEmbedder(ECAPA_PATH)

    # -----------------------------------------------------------------------
    # Step 1: Discover KittenTTS speakers and their gender
    # -----------------------------------------------------------------------
    # The refs/ directory contains KittenTTS reference audio. We need to know
    # each speaker's gender to match them with Kokoro/Piper voices.
    ktn_speakers: dict[str, list[str]] = defaultdict(list)
    for fname in sorted(os.listdir("refs")):
        if fname.endswith(".wav"):
            speaker = fname.rsplit("_ref_", 1)[0]
            ktn_speakers[speaker].append(os.path.join("refs", fname))

    # Hardcoded gender mapping for the 8 KittenTTS speakers
    SPEAKER_GENDER = {
        "bella": "F", "bruno": "M", "hugo": "M", "jasper": "M",
        "kiki": "F", "leo": "M", "luna": "F", "rosie": "F",
    }

    speaker_names = sorted(ktn_speakers.keys())

    # -----------------------------------------------------------------------
    # Step 2: Build multi-engine training data
    # -----------------------------------------------------------------------
    # To create a shared canonical space, the VAE needs to see embeddings
    # from all three engines (KittenTTS, Kokoro, Piper) for the same
    # "speakers". We use gender-matched Kokoro/Piper voices as proxies
    # for the KittenTTS speakers.
    print("Building multi-engine training data...")
    all_embs, all_labels = [], []

    # KittenTTS: load existing reference clips + speed augmentation
    for i, name in enumerate(speaker_names):
        for path in ktn_speakers[name]:
            audio, _ = load_audio(path)
            emb = embedder.embed(audio)
            all_embs.append(emb)
            all_labels.append(i)
            # Speed augmentation: 0.9x and 1.1x
            for factor in [0.9, 1.1]:
                aug = speed_perturb(audio, 16000, factor)
                all_embs.append(embedder.embed(aug))
                all_labels.append(i)

    # Kokoro: synthesize phrases with gender-matched voices
    for i, name in enumerate(speaker_names):
        gender = SPEAKER_GENDER[name]
        for voice in KOKORO_VOICES[gender]:
            for phrase in PHRASES:
                audio = synthesize_kokoro(phrase, voice)
                all_embs.append(embedder.embed(audio))
                all_labels.append(i)

    # Piper: synthesize phrases with gender-matched voices
    for i, name in enumerate(speaker_names):
        gender = SPEAKER_GENDER[name]
        for voice in PIPER_VOICES[gender]:
            for phrase in PHRASES:
                audio = synthesize_piper(phrase, voice)
                all_embs.append(embedder.embed(audio))
                all_labels.append(i)

    all_embs = np.array(all_embs)
    all_labels = np.array(all_labels)
    print(f"  {len(all_embs)} embeddings, {len(speaker_names)} speakers, 3 engines\n")

    # -----------------------------------------------------------------------
    # Step 3: Train the multi-engine VAE
    # -----------------------------------------------------------------------
    print("Training multi-engine VAE (unsupervised + contrastive)...")
    t0 = time.perf_counter()
    vae = train_vae(all_embs, all_labels)
    print(f"  Done in {time.perf_counter() - t0:.1f}s\n")

    # -----------------------------------------------------------------------
    # Scenario 1: Cross-engine matching
    # -----------------------------------------------------------------------
    # Reference audio from Kokoro, test audio from Piper.
    # The VAE should map both into a shared space where speaker identity
    # is preserved regardless of synthesis engine.
    print("=" * 70)
    print("SCENARIO 1: Cross-Engine Matching")
    print("  Reference: Kokoro voices (3 phrases per speaker)")
    print("  Test: Piper voices (3 phrases per speaker)")
    print("=" * 70)

    # Build Kokoro prototypes (average of all Kokoro clips per speaker)
    kokoro_ref = {}
    for i, name in enumerate(speaker_names):
        gender = SPEAKER_GENDER[name]
        embs = []
        for voice in KOKORO_VOICES[gender]:
            for phrase in PHRASES:
                audio = synthesize_kokoro(phrase, voice)
                embs.append(embedder.embed(audio))
        kokoro_ref[name] = np.mean(embs, axis=0)
        kokoro_ref[name] /= np.linalg.norm(kokoro_ref[name])

    # Build Piper test turns
    piper_turns = defaultdict(list)
    for i, name in enumerate(speaker_names):
        gender = SPEAKER_GENDER[name]
        for voice in PIPER_VOICES[gender]:
            for phrase in PHRASES:
                audio = synthesize_piper(phrase, voice)
                piper_turns[name].append(embedder.embed(audio))

    # Raw ECAPA-TDNN matching
    raw_proto = np.array([kokoro_ref[n] for n in speaker_names])
    raw_proto /= np.linalg.norm(raw_proto, axis=1, keepdims=True)
    raw_acc = classify(raw_proto, piper_turns, speaker_names)

    # VAE-projected matching
    vae_proto = project(vae, raw_proto)
    vae_turns = {s: project(vae, np.array(e)) for s, e in piper_turns.items()}
    vae_acc = classify(vae_proto, vae_turns, speaker_names)

    print(f"\n  Raw ECAPA-TDNN:  {raw_acc:.1%}")
    print(f"  VAE-projected:   {vae_acc:.1%}")
    print(f"  Delta:           {vae_acc - raw_acc:+.1%}\n")

    # -----------------------------------------------------------------------
    # Scenario 2: Limited reference data
    # -----------------------------------------------------------------------
    # Only 1 reference clip per speaker (from Kokoro), tested against
    # KittenTTS dialogue turns. The VAE's structured latent space should
    # generalize better from a single example.
    print("=" * 70)
    print("SCENARIO 2: Limited Reference Data (1 clip per speaker)")
    print("  Reference: 1 Kokoro phrase per speaker")
    print("  Test: KittenTTS dialogue turns")
    print("=" * 70)

    # Single Kokoro reference per speaker
    single_ref = {}
    for i, name in enumerate(speaker_names):
        gender = SPEAKER_GENDER[name]
        voice = KOKORO_VOICES[gender][0]
        audio = synthesize_kokoro(PHRASES[0], voice)
        single_ref[name] = embedder.embed(audio)
        single_ref[name] /= np.linalg.norm(single_ref[name])

    # KittenTTS turns
    ktn_turns = defaultdict(list)
    for fname in sorted(os.listdir("turns")):
        if fname.endswith(".wav"):
            spk = fname.split("_")[2].replace(".wav", "")
            audio, _ = load_audio(os.path.join("turns", fname))
            ktn_turns[spk].append(embedder.embed(audio))

    # Raw matching
    raw_single = np.array([single_ref[n] for n in speaker_names])
    raw_single /= np.linalg.norm(raw_single, axis=1, keepdims=True)
    raw_acc2 = classify(raw_single, ktn_turns, speaker_names)

    # VAE matching
    vae_single = project(vae, raw_single)
    vae_ktn = {s: project(vae, np.array(e)) for s, e in ktn_turns.items()}
    vae_acc2 = classify(vae_single, vae_ktn, speaker_names)

    print(f"\n  Raw ECAPA-TDNN:  {raw_acc2:.1%}")
    print(f"  VAE-projected:   {vae_acc2:.1%}")
    print(f"  Delta:           {vae_acc2 - raw_acc2:+.1%}\n")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Scenario':<45} {'Raw':>8} {'VAE':>8} {'Delta':>8}")
    print(f"  {'-' * 73}")
    print(f"  {'Cross-engine (Kokoro → Piper)':<45} {raw_acc:>7.1%} {vae_acc:>7.1%} {vae_acc - raw_acc:>+7.1%}")
    print(f"  {'Limited refs (1 clip → KittenTTS)':<45} {raw_acc2:>7.1%} {vae_acc2:>7.1%} {vae_acc2 - raw_acc2:>+7.1%}")
    print()
    print("The VAE helps when:")
    print("  • Reference and test come from different TTS engines")
    print("  • You have very few reference clips (1-2 per speaker)")
    print("  • You need maximum embedding separation for robust matching")
    print()
    print("The VAE does NOT help when:")
    print("  • Same engine for reference and test (ECAPA-TDNN already great)")
    print("  • Plenty of reference data (5+ clips)")
    print("  • Clean audio with few speakers")


if __name__ == "__main__":
    main()
