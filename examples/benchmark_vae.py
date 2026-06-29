"""VAE Benchmark: Comprehensive evaluation of all training strategies.

Tests all combinations of:
- Latent dimensions: 32, 64, 128
- KL free-bits: 0 (standard), 2.0
- Training: unsupervised, contrastive FT, prototypical FT
- Tasks: speaker identification, diarization

Results are displayed in a clear table format for easy comparison.
"""

import os
import time
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from kittentts import KittenTTS

from speech_segmentation import Diarizer, SpeakerEmbedder, SpeechSegmenter
from speech_segmentation.augmentation import speed_perturb
from speech_segmentation.vae import (
    VAE,
    SpeakerVAE,
    prototypical_loss,
    supervised_contrastive_loss,
)

EMB_MODEL_PATH = "models/ecapa_tdnn.onnx"
SEG_MODEL_PATH = "models/model.onnx"
TTS_SR = 24000
SILENCE_GAP_SEC = 0.5

REFERENCE_PHRASES = {
    "Bella": [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "The early bird catches the worm but the second mouse gets the cheese.",
        "To be or not to be that is the question.",
        "All that glitters is not gold but silver shines too.",
    ],
    "Bruno": [
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump.",
        "The five boxing wizards jump quickly across the mat.",
        "Actions speak louder than words but silence speaks volumes.",
        "The pen is mightier than the sword when wielded well.",
    ],
    "Luna": [
        "She sells seashells by the seashore every Sunday morning.",
        "Peter Piper picked a peck of pickled peppers.",
        "Unique New York, unique New York, you know you need unique New York.",
        "Fortune favors the prepared mind in every endeavor.",
        "Curiosity killed the cat but satisfaction brought it back.",
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
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be that is the question.",
    "All that glitters is not gold but silver shines too.",
    "Actions speak louder than words but silence speaks volumes.",
    "The pen is mightier than the sword when wielded well.",
    "Fortune favors the prepared mind in every endeavor.",
    "Curiosity killed the cat but satisfaction brought it back.",
    "Early to bed and early to rise makes a person healthy.",
    "Where there is a will there is always a way forward.",
    "A picture is worth a thousand words in every language.",
    "The early bird catches the worm but the second mouse gets cheese.",
    "Two wrongs never make a right but three lefts do always.",
    "A watched pot never boils but an unwatched one boils over.",
    "Birds of a feather flock together in every season.",
    "The grass is always greener on the other side of fence.",
    "When in Rome do as the Romans do every single day.",
    "You can judge a book by looking at its cover sometimes.",
    "The apple does not fall far from the tree as they say.",
    "Every cloud has a silver lining if you look closely.",
    "Not all those who wander are lost but some are just exploring.",
    "A watched pot never boils but patience makes it easier.",
    "The pen is mightier but the keyboard is more convenient.",
    "A journey of a thousand steps begins with finding your shoes.",
    "The best time to plant a tree was twenty years ago.",
    "Knowledge is power but enthusiasm pulls the switch.",
    "Actions speak louder than words but silence speaks volumes.",
    "Fortune favors the prepared mind in every endeavor.",
    "Curiosity killed the cat but satisfaction brought it back.",
]

UNLABELLED_VOICES = ["Bella", "Bruno", "Luna", "Hugo", "Rosie", "Leo", "Jasper", "Kiki"]


@dataclass
class BenchmarkResult:
    variant: str
    latent_dim: int
    free_bits: float
    speaker_id_acc: float = 0.0
    diarization_acc: float = 0.0
    train_time: float = 0.0
    details: dict = field(default_factory=dict)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_reference_samples(tts, ref_dir="refs_bench"):
    os.makedirs(ref_dir, exist_ok=True)
    ref_paths = {}
    for voice, phrases in REFERENCE_PHRASES.items():
        paths = [os.path.join(ref_dir, f"{voice.lower()}_ref_{j}.wav") for j in range(len(phrases))]
        ref_paths[voice] = paths
        if all(os.path.exists(p) for p in paths):
            continue
        for j, phrase in enumerate(phrases):
            audio = tts.generate(phrase, voice=voice)
            sf.write(paths[j], audio, TTS_SR)
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
        avg_emb = np.mean(voice_embs, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        ref_embeddings[voice] = avg_emb
    return ref_embeddings


def generate_unlabelled_samples(tts, unlabelled_dir="unlabelled_bench"):
    os.makedirs(unlabelled_dir, exist_ok=True)
    unlabelled_paths = []
    for i, phrase in enumerate(UNLABELLED_PHRASES):
        path = os.path.join(unlabelled_dir, f"unlabelled_{i:02d}.wav")
        if os.path.exists(path):
            unlabelled_paths.append(path)
            continue
        voice = UNLABELLED_VOICES[i % len(UNLABELLED_VOICES)]
        audio = tts.generate(phrase, voice=voice)
        sf.write(path, audio, TTS_SR)
        unlabelled_paths.append(path)

        for factor in [0.9, 1.1]:
            aug_path = os.path.join(unlabelled_dir, f"unlabelled_{i:02d}_speed{factor:.1f}.wav")
            if not os.path.exists(aug_path):
                aug_audio = speed_perturb(audio, TTS_SR, factor)
                sf.write(aug_path, aug_audio, TTS_SR)
            unlabelled_paths.append(aug_path)
    return unlabelled_paths


def collect_segment_embeddings(unlabelled_paths, embedder, segmenter):
    FRAME_STEP = 270
    MIN_EMBED_SAMPLES = 12800

    turn_paths = (
        sorted(os.path.join("turns_bench", f) for f in os.listdir("turns_bench") if f.endswith(".wav"))
        if os.path.isdir("turns_bench")
        else []
    )

    all_paths = list(unlabelled_paths) + turn_paths
    silence_samples = int(SILENCE_GAP_SEC * TTS_SR)
    silence = np.zeros(silence_samples, dtype=np.float32)
    chunks = []
    for i, path in enumerate(all_paths):
        if not os.path.exists(path):
            continue
        audio, _ = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        chunks.append(audio)
        if i < len(all_paths) - 1:
            chunks.append(silence)
    combined_tts = np.concatenate(chunks)

    combined_16k = np.interp(
        np.linspace(0, len(combined_tts) - 1, int(len(combined_tts) * 16000 / TTS_SR)),
        np.arange(len(combined_tts)),
        combined_tts,
    ).astype(np.float32)

    segments = segmenter.segment(combined_16k)
    long_segs = [s for s in segments if (s.end_frame - s.start_frame) * FRAME_STEP >= MIN_EMBED_SAMPLES]

    embeddings = []
    for seg in long_segs:
        start_sample = seg.start_frame * FRAME_STEP
        end_sample = seg.end_frame * FRAME_STEP
        seg_audio = combined_16k[start_sample:end_sample]
        emb = embedder.embed(seg_audio)
        embeddings.append(emb)

    return np.array(embeddings, dtype=np.float32)


def collect_clean_embeddings(ref_paths, embedder):
    embeddings = []
    for voice, paths in ref_paths.items():
        for path in paths:
            audio, _ = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            emb = embedder.embed(audio)
            embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)


def collect_labelled_embeddings(ref_paths, embedder):
    embeddings = []
    labels = []
    for label_idx, (voice, paths) in enumerate(ref_paths.items()):
        for path in paths:
            audio, _ = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            emb = embedder.embed(audio)
            embeddings.append(emb)
            labels.append(label_idx)
    return np.array(embeddings, dtype=np.float32), np.array(labels, dtype=np.int64)


def generate_dialogue_turns(tts, output_dir="turns_bench"):
    os.makedirs(output_dir, exist_ok=True)
    turn_paths = []
    for i, (voice, text) in enumerate(DIALOGUE):
        path = os.path.join(output_dir, f"turn_{i:02d}_{voice.lower()}.wav")
        if os.path.exists(path):
            turn_paths.append(path)
            continue
        audio = tts.generate(text, voice=voice)
        sf.write(path, audio, TTS_SR)
        turn_paths.append(path)
    return turn_paths


def classify_turns(turn_paths, embedder, ref_embeddings, vae=None):
    ref_names = list(ref_embeddings.keys())
    ref_matrix = np.array([ref_embeddings[n] for n in ref_names])
    if vae is not None:
        ref_matrix = vae.encode_batch(ref_matrix)

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
        if ref_names[best_idx] == gt_voice:
            correct += 1
    return correct, len(turn_paths)


def evaluate_diarization(matches, ground_truth):
    all_speakers = sorted(set(s for s, _, _ in ground_truth))
    correct = 0
    total = len(matches)
    for m in matches:
        overlaps = {}
        for speaker in all_speakers:
            total_overlap = sum(
                max(0.0, min(m.end_time, gt_end) - max(m.start_time, gt_start))
                for s, gt_start, gt_end in ground_truth
                if s == speaker
            )
            overlaps[speaker] = total_overlap
        expected = max(overlaps, key=overlaps.get)
        if m.speaker == expected:
            correct += 1
    return correct, total


def run_diarization(segmenter, embedder, ref_embeddings, turn_paths, vae=None):
    FRAME_STEP = 270
    tts_sr = TTS_SR

    silence_samples = int(SILENCE_GAP_SEC * tts_sr)
    silence = np.zeros(silence_samples, dtype=np.float32)
    chunks = []
    ground_truth = []
    cursor_sec = 0.0

    for i, path in enumerate(turn_paths):
        audio, _ = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        chunks.append(audio)
        turn_duration = len(audio) / tts_sr
        speaker = DIALOGUE[i][0]
        ground_truth.append((speaker, cursor_sec, cursor_sec + turn_duration))
        cursor_sec += turn_duration
        if i < len(turn_paths) - 1:
            chunks.append(silence)
            cursor_sec += SILENCE_GAP_SEC

    combined_tts = np.concatenate(chunks)
    combined_16k = np.interp(
        np.linspace(0, len(combined_tts) - 1, int(len(combined_tts) * 16000 / tts_sr)),
        np.arange(len(combined_tts)),
        combined_tts,
    ).astype(np.float32)

    diarizer = Diarizer(segmenter, embedder, vae=vae)
    diarizer.build_references(ref_embeddings)
    matches, _ = diarizer.diarize(combined_16k)
    return evaluate_diarization(matches, ground_truth)


def train_vae_unsupervised(embeddings, latent_dim, free_bits, epochs=500, seed=42):
    set_seed(seed)
    model = VAE(192, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    data = torch.from_numpy(embeddings)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=min(32, len(data)),
        shuffle=True,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        beta = min(1.0, epoch / 100)
        for (batch_x,) in loader:
            noisy = batch_x + torch.randn_like(batch_x) * 0.05
            recon, mu, logvar = model(noisy)
            recon_loss = F.mse_loss(recon, batch_x)
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            if free_bits > 0:
                kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
            kl_loss = kl_per_dim.mean()
            loss = recon_loss + beta * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def train_vae_contrastive(unlabelled_embs, labelled_embs, labelled_labels,
                          latent_dim, free_bits, pretrain_epochs=300,
                          finetune_epochs=200, contrastive_temp=0.1, seed=42):
    set_seed(seed)
    model = VAE(192, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    data = torch.from_numpy(unlabelled_embs)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=min(32, len(data)),
        shuffle=True,
    )

    model.train()
    for epoch in range(1, pretrain_epochs + 1):
        beta = min(1.0, epoch / 100)
        for (batch_x,) in loader:
            noisy = batch_x + torch.randn_like(batch_x) * 0.05
            recon, mu, logvar = model(noisy)
            recon_loss = F.mse_loss(recon, batch_x)
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            if free_bits > 0:
                kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
            kl_loss = kl_per_dim.mean()
            loss = recon_loss + beta * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    labelled_data = torch.from_numpy(labelled_embs).float()
    labelled_labels_t = torch.from_numpy(labelled_labels).long()
    ft_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(labelled_data, labelled_labels_t),
        batch_size=min(32, len(labelled_data)),
        shuffle=True,
    )

    optimizer_ft = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=finetune_epochs)

    model.train()
    for epoch in range(1, finetune_epochs + 1):
        for batch_x, batch_labels in ft_loader:
            noisy = batch_x + torch.randn_like(batch_x) * 0.05
            _, mu, logvar = model(noisy)
            z = mu / (mu.norm(dim=1, keepdim=True) + 1e-8)
            loss_con = supervised_contrastive_loss(z, batch_labels, temperature=contrastive_temp)
            recon_loss = F.mse_loss(model.decode(mu), batch_x)
            loss = 0.8 * recon_loss + 0.2 * loss_con
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()
        scheduler.step()

    return model


def model_to_vae(model, latent_dim):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(
            {"_vae_config": {"input_dim": 192, "latent_dim": latent_dim}, "state_dict": model.state_dict()},
            f.name,
        )
        return SpeakerVAE(f.name)


def run_benchmark():
    print("=" * 80)
    print("  VAE BENCHMARK: Speaker Identification & Diarization")
    print("=" * 80)

    tts = KittenTTS("KittenML/kitten-tts-nano-0.8")
    embedder = SpeakerEmbedder(EMB_MODEL_PATH)
    segmenter = SpeechSegmenter(SEG_MODEL_PATH)

    print("\n--- Data Preparation ---")
    ref_paths = generate_reference_samples(tts)
    ref_embeddings = build_reference_embeddings(ref_paths, embedder)
    unlabelled_paths = generate_unlabelled_samples(tts)
    segment_embs = collect_segment_embeddings(unlabelled_paths, embedder, segmenter)
    clean_embs = collect_clean_embeddings(ref_paths, embedder)
    unlabelled_embs = np.concatenate([segment_embs, clean_embs], axis=0)
    labelled_embs, labelled_labels = collect_labelled_embeddings(ref_paths, embedder)
    turn_paths = generate_dialogue_turns(tts)

    print(f"  Unlabelled embeddings: {len(unlabelled_embs)}")
    print(f"  Labelled embeddings: {len(labelled_embs)} ({len(np.unique(labelled_labels))} speakers)")
    print(f"  Dialogue turns: {len(turn_paths)}")

    no_vae_sid_correct, no_vae_sid_total = classify_turns(turn_paths, embedder, ref_embeddings, vae=None)
    no_vae_diar_correct, no_vae_diar_total = run_diarization(segmenter, embedder, ref_embeddings, turn_paths, vae=None)

    configs = [
        (32, 0.0),
        (32, 2.0),
        (64, 0.0),
        (64, 2.0),
        (128, 0.0),
        (128, 2.0),
    ]

    results = []

    results.append(BenchmarkResult(
        variant="No VAE (baseline)",
        latent_dim=0,
        free_bits=0,
        speaker_id_acc=no_vae_sid_correct / no_vae_sid_total,
        diarization_acc=no_vae_diar_correct / no_vae_diar_total,
    ))

    for latent_dim, free_bits in configs:
        for variant_name, variant_fn in [
            ("Unsupervised", lambda: train_vae_unsupervised(unlabelled_embs, latent_dim, free_bits)),
            ("Contrastive FT", lambda: train_vae_contrastive(unlabelled_embs, labelled_embs, labelled_labels, latent_dim, free_bits)),
        ]:
            print(f"\n--- {variant_name} | dim={latent_dim} | fb={free_bits} ---")

            start = time.time()
            model = variant_fn()
            train_time = time.time() - start

            vae = model_to_vae(model, latent_dim)

            sid_correct, sid_total = classify_turns(turn_paths, embedder, ref_embeddings, vae=vae)
            diar_correct, diar_total = run_diarization(segmenter, embedder, ref_embeddings, turn_paths, vae=vae)

            result = BenchmarkResult(
                variant=variant_name,
                latent_dim=latent_dim,
                free_bits=free_bits,
                speaker_id_acc=sid_correct / sid_total,
                diarization_acc=diar_correct / diar_total,
                train_time=train_time,
                details={"sid": f"{sid_correct}/{sid_total}", "diar": f"{diar_correct}/{diar_total}"},
            )
            results.append(result)

            print(f"  Speaker ID: {sid_correct}/{sid_total} ({result.speaker_id_acc * 100:.1f}%)")
            print(f"  Diarization: {diar_correct}/{diar_total} ({result.diarization_acc * 100:.1f}%)")
            print(f"  Train time: {train_time:.1f}s")

    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Variant':<20s} {'Dim':>4s} {'FB':>5s} {'Speaker ID':>12s} {'Diarization':>12s} {'Time':>7s}")
    print("-" * 65)
    for r in results:
        sid_str = r.details.get("sid", "—")
        diar_str = r.details.get("diar", "—")
        print(f"{r.variant:<20s} {r.latent_dim:>4d} {r.free_bits:>5.1f} {r.speaker_id_acc * 100:>10.1f}% {r.diarization_acc * 100:>10.1f}% {r.train_time:>6.1f}s")

    best_sid = max(results, key=lambda r: r.speaker_id_acc)
    best_diar = max(results, key=lambda r: r.diarization_acc)

    print(f"\n  Best Speaker ID:    {best_sid.variant} (dim={best_sid.latent_dim}, fb={best_sid.free_bits}) = {best_sid.speaker_id_acc * 100:.1f}%")
    print(f"  Best Diarization:   {best_diar.variant} (dim={best_diar.latent_dim}, fb={best_diar.free_bits}) = {best_diar.diarization_acc * 100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
