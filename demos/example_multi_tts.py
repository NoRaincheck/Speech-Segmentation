"""Multi-TTS embedding test: does VAE improve cross-engine results?

Synthesizes the same phrase with Kokoro-ONNX and Piper-TTS voices, then
compares ECAPA-TDNN raw embeddings vs VAE-projected embeddings. If VAE
removes engine-specific variance, cross-engine same-gender similarity
should increase after projection.

Usage:
    uv run --group vae python demos/example_multi_tts.py
"""

import os
import time

import numpy as np
import torch

from speech_segmentation import SpeakerEmbedder
from speech_segmentation.audio import load_audio
from speech_segmentation.augmentation import speed_perturb
from speech_segmentation.vae import VAE, supervised_contrastive_loss

MODELS_DIR = "models"
PHRASE = "The quick brown fox jumps over the lazy dog."


def synthesize_kokoro(text: str, voice: str) -> np.ndarray:
    from kokoro_onnx import Kokoro

    k = Kokoro(f"{MODELS_DIR}/kokoro-v1.0.onnx", f"{MODELS_DIR}/voices-v1.0.bin")
    audio, sr = k.create(text, voice=voice, speed=1.0)
    if sr != 16000:
        indices = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    return audio.astype(np.float32)


def synthesize_piper(text: str, model_name: str) -> np.ndarray:
    from piper import PiperVoice

    v = PiperVoice.load(f"{MODELS_DIR}/{model_name}.onnx")
    chunks = list(v.synthesize(text))
    audio = np.concatenate([c.audio_float_array for c in chunks])
    sr = v.config.sample_rate
    if sr != 16000:
        indices = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    return audio.astype(np.float32)


def train_vae(train_embs: np.ndarray, train_labels: np.ndarray, rng: np.random.RandomState) -> VAE:
    """Train VAE with contrastive fine-tuning on KittenTTS reference data."""
    n_speakers = len(np.unique(train_labels))
    input_dim = train_embs.shape[1]
    latent_dim = 64

    # Unsupervised pre-training
    model = VAE(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(train_embs).float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=32, shuffle=True)

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

    # Contrastive fine-tuning
    model_ft = VAE(input_dim, latent_dim, num_speakers=n_speakers)
    model_ft.load_state_dict(model.state_dict(), strict=False)
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-4)
    y = torch.from_numpy(train_labels).long()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=32, shuffle=True)
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


def analyze(sim_matrix, engines, labels, descs, short, label):
    """Print similarity matrix and analysis."""
    n = len(engines)

    print(f"\n  {label} — pairwise cosine similarity:")
    print()
    print(f"{'':>10}", end="")
    for s in short:
        print(f"{s:>8}", end="")
    print()
    print("-" * (10 + 8 * n))
    for i in range(n):
        print(f"{short[i]:>10}", end="")
        for j in range(n):
            print(f"{sim_matrix[i, j]:>8.3f}", end="")
        print()

    def get_gender(label):
        if any(x in label for x in ["adam", "michael", "george", "lessac", "male"]):
            return "M"
        if any(x in label for x in ["bella", "sarah", "emma", "amy", "female"]):
            return "F"
        return "?"

    same_engine, diff_engine = [], []
    same_gender, diff_gender = [], []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if engines[i] == engines[j]:
                same_engine.append(sim)
            else:
                diff_engine.append(sim)
            gi, gj = get_gender(labels[i]), get_gender(labels[j])
            if gi == gj and gi != "?":
                same_gender.append(sim)
            elif gi != "?" and gj != "?":
                diff_gender.append(sim)

    print(f"\n  Same engine, same gender:  {np.mean(same_engine):.4f}")
    print(f"  Diff engine, same gender:  {np.mean(diff_engine):.4f}")
    print(f"  Same engine, diff gender:  {np.mean(same_gender):.4f}")
    print(f"  Diff engine, diff gender:  {np.mean(diff_gender):.4f}")
    print(f"  Engine premium:            {np.mean(same_engine) - np.mean(diff_engine):+.4f}")

    return {
        "same_engine": np.mean(same_engine),
        "diff_engine": np.mean(diff_engine),
        "same_gender": np.mean(same_gender),
        "diff_gender": np.mean(diff_gender),
    }


def main():
    print("=" * 70)
    print("MULTI-TTS TEST: Does VAE Improve Cross-Engine Results?")
    print("=" * 70)
    print()
    print("If VAE removes TTS engine variance while preserving speaker signal,")
    print("cross-engine same-gender similarity should INCREASE after projection.")
    print()

    embedder = SpeakerEmbedder(f"{MODELS_DIR}/ecapa_tdnn.onnx")

    # Synthesize voices
    voices = []
    kokoro_voices = [
        ("am_adam", "am_adam", "Kokoro Adam"),
        ("am_michael", "am_michael", "Kokoro Michael"),
        ("af_bella", "af_bella", "Kokoro Bella"),
        ("af_sarah", "af_sarah", "Kokoro Sarah"),
        ("bm_george", "bm_george", "Kokoro George"),
        ("bf_emma", "bf_emma", "Kokoro Emma"),
    ]
    print("Synthesizing Kokoro voices...")
    for label, voice_id, desc in kokoro_voices:
        audio = synthesize_kokoro(PHRASE, voice_id)
        emb = embedder.embed(audio)
        voices.append((label, "Kokoro", desc, emb))

    piper_voices = [
        ("lessac", "en_US-lessac-medium", "Piper Lessac"),
        ("amy", "en_US-amy-medium", "Piper Amy"),
    ]
    print("Synthesizing Piper voices...")
    for label, model, desc in piper_voices:
        audio = synthesize_piper(PHRASE, model)
        emb = embedder.embed(audio)
        voices.append((label, "Piper", desc, emb))

    n = len(voices)
    labels = [v[0] for v in voices]
    engines = [v[1] for v in voices]
    descs = [v[2] for v in voices]
    embeddings = np.array([v[3] for v in voices])
    short = ["K-adam", "K-mich", "K-bell", "K-sara", "K-geo", "K-emm", "P-less", "P-amy"]

    # --- Raw ECAPA-TDNN ---
    raw_sims = embeddings @ embeddings.T
    raw_stats = analyze(raw_sims, engines, labels, descs, short, "RAW ECAPA-TDNN")

    # --- Train VAE on KittenTTS references ---
    print("\nTraining VAE on KittenTTS reference data...")
    speaker_files = {}
    for fname in sorted(os.listdir("refs")):
        if fname.endswith(".wav"):
            speaker = fname.rsplit("_ref_", 1)[0]
            speaker_files.setdefault(speaker, []).append(os.path.join("refs", fname))

    train_embs, train_labels = [], []
    all_names = sorted(speaker_files.keys())
    for i, name in enumerate(all_names):
        for p in speaker_files[name]:
            audio, _ = load_audio(p)
            e = embedder.embed(audio)
            train_embs.append(e)
            train_labels.append(i)
            for f in [0.9, 1.1]:
                train_embs.append(embedder.embed(speed_perturb(audio, 16000, f)))
                train_labels.append(i)

    train_embs = np.array(train_embs)
    train_labels = np.array(train_labels)

    t0 = time.perf_counter()
    vae = train_vae(train_embs, train_labels, np.random.RandomState(42))
    print(f"  Trained in {time.perf_counter() - t0:.1f}s")

    # Project multi-TTS embeddings through VAE
    def project(embs_arr):
        t = torch.from_numpy(embs_arr).float()
        with torch.no_grad():
            mu, _ = vae.encode(t)
        z = mu.numpy()
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        return z / norms

    vae_embeddings = project(embeddings)
    vae_sims = vae_embeddings @ vae_embeddings.T
    vae_stats = analyze(vae_sims, engines, labels, descs, short, "VAE-PROJECTED (contrastive FT)")

    # --- Comparison ---
    print()
    print("=" * 60)
    print("COMPARISON: Raw vs VAE")
    print("=" * 60)
    print()
    print(f"{'Metric':<30} {'Raw':>8} {'VAE':>8} {'Delta':>8}")
    print("-" * 58)
    for key in ["same_engine", "diff_engine", "same_gender", "diff_gender"]:
        r = raw_stats[key]
        v = vae_stats[key]
        d = v - r
        print(f"  {key:<28} {r:>7.4f} {v:>7.4f} {d:>+7.4f}")

    raw_engine_premium = raw_stats["same_engine"] - raw_stats["diff_engine"]
    vae_engine_premium = vae_stats["same_engine"] - vae_stats["diff_engine"]
    print(f"  {'engine_premium':<28} {raw_engine_premium:>7.4f} {vae_engine_premium:>7.4f} {vae_engine_premium - raw_engine_premium:>+7.4f}")

    print()
    cross_gender_delta = (vae_stats["diff_engine"] - raw_stats["diff_engine"])
    engine_delta = vae_engine_premium - raw_engine_premium
    if engine_delta < 0:
        print(f"  VAE REDUCES engine premium by {abs(engine_delta):.4f}")
        print(f"  Cross-engine similarity {'improves' if cross_gender_delta > 0 else 'changes'}: {cross_gender_delta:+.4f}")
    else:
        print(f"  VAE INCREASES engine premium by {engine_delta:.4f}")
        print(f"  Cross-engine similarity change: {cross_gender_delta:+.4f}")

    print()
    print("Cross-engine same-gender pairs:")
    print(f"  {'Pair':<40} {'Raw':>8} {'VAE':>8} {'Delta':>8}")
    print("  " + "-" * 68)
    for i in range(n):
        for j in range(i + 1, n):
            if engines[i] != engines[j]:
                gi = "M" if any(x in labels[i] for x in ["adam", "michael", "george", "lessac"]) else "F"
                gj = "M" if any(x in labels[j] for x in ["adam", "michael", "george", "lessac"]) else "F"
                if gi == gj:
                    r = float(raw_sims[i, j])
                    v = float(vae_sims[i, j])
                    print(f"  {descs[i]:<18} <-> {descs[j]:<18} {r:>7.4f} {v:>7.4f} {v - r:>+7.4f}")


if __name__ == "__main__":
    main()
