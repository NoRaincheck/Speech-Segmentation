"""Ablation study for VAE training strategies.

Compares different training approaches:
1. Unsupervised VAE (reconstruction + KL only)
2. VAE + contrastive fine-tuning
3. VAE + prototypical loss fine-tuning
4. VAE + combined loss (reconstruction + contrastive)

Each variant is trained from scratch and evaluated on speaker identification.
"""

import os
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from kittentts import KittenTTS

from speech_segmentation import SpeakerEmbedder, SpeechSegmenter
from speech_segmentation.augmentation import speed_perturb
from speech_segmentation.vae import (
    VAE,
    SpeakerVAE,
    prototypical_loss,
    supervised_contrastive_loss,
)

EMB_MODEL_PATH = "models/ecapa_tdnn.onnx"
VAE_INPUT_DIM = 192
VAE_LATENT_DIM = 64
VAE_NOISE_STD = 0.05
VAE_BATCH_SIZE = 32
VAE_LR = 1e-3
VAE_KL_WARMUP_EPOCHS = 100
VAE_KL_FREE_BITS = 2.0

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


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_reference_samples(tts, ref_dir="refs_ablation"):
    os.makedirs(ref_dir, exist_ok=True)
    ref_paths = {}
    for voice, phrases in REFERENCE_PHRASES.items():
        paths = [os.path.join(ref_dir, f"{voice.lower()}_ref_{j}.wav") for j in range(len(phrases))]
        ref_paths[voice] = paths
        if all(os.path.exists(p) for p in paths):
            continue
        for j, phrase in enumerate(phrases):
            audio = tts.generate(phrase, voice=voice)
            sf.write(paths[j], audio, 24000)
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


def generate_unlabelled_samples(tts, unlabelled_dir="unlabelled_ablation"):
    os.makedirs(unlabelled_dir, exist_ok=True)
    unlabelled_paths = []
    for i, phrase in enumerate(UNLABELLED_PHRASES):
        path = os.path.join(unlabelled_dir, f"unlabelled_{i:02d}.wav")
        if os.path.exists(path):
            unlabelled_paths.append(path)
            continue
        voice = UNLABELLED_VOICES[i % len(UNLABELLED_VOICES)]
        audio = tts.generate(phrase, voice=voice)
        sf.write(path, audio, 24000)
        unlabelled_paths.append(path)

        for factor in [0.9, 1.1]:
            aug_path = os.path.join(unlabelled_dir, f"unlabelled_{i:02d}_speed{factor:.1f}.wav")
            if not os.path.exists(aug_path):
                aug_audio = speed_perturb(audio, 24000, factor)
                sf.write(aug_path, aug_audio, 24000)
            unlabelled_paths.append(aug_path)
    return unlabelled_paths


def collect_segment_embeddings(unlabelled_paths, embedder, segmenter):
    FRAME_STEP = 270
    MIN_EMBED_SAMPLES = 12800

    turn_paths = (
        sorted(os.path.join("turns_ablation", f) for f in os.listdir("turns_ablation") if f.endswith(".wav"))
        if os.path.isdir("turns_ablation")
        else []
    )

    all_paths = list(unlabelled_paths) + turn_paths
    silence_samples = int(0.5 * 24000)
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
        np.linspace(0, len(combined_tts) - 1, int(len(combined_tts) * 16000 / 24000)),
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
    label_names = list(ref_paths.keys())
    for label_idx, (voice, paths) in enumerate(ref_paths.items()):
        for path in paths:
            audio, _ = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            emb = embedder.embed(audio)
            embeddings.append(emb)
            labels.append(label_idx)
    return np.array(embeddings, dtype=np.float32), np.array(labels, dtype=np.int64)


def generate_dialogue_turns(tts, output_dir="turns_ablation"):
    os.makedirs(output_dir, exist_ok=True)
    turn_paths = []
    for i, (voice, text) in enumerate(DIALOGUE):
        path = os.path.join(output_dir, f"turn_{i:02d}_{voice.lower()}.wav")
        if os.path.exists(path):
            turn_paths.append(path)
            continue
        audio = tts.generate(text, voice=voice)
        sf.write(path, audio, 24000)
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
        pred = ref_names[best_idx]
        if pred == gt_voice:
            correct += 1
    return correct, len(turn_paths)


def train_vae_unsupervised(embeddings, save_path, epochs=500, seed=42):
    set_seed(seed)
    model = VAE(VAE_INPUT_DIM, VAE_LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_LR, weight_decay=1e-4)

    data = torch.from_numpy(embeddings)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=min(VAE_BATCH_SIZE, len(data)),
        shuffle=True,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        beta = min(1.0, epoch / VAE_KL_WARMUP_EPOCHS)
        for (batch_x,) in loader:
            noisy = batch_x + torch.randn_like(batch_x) * VAE_NOISE_STD
            recon, mu, logvar = model(noisy)
            recon_loss = F.mse_loss(recon, batch_x)
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_per_dim = torch.clamp(kl_per_dim, min=VAE_KL_FREE_BITS)
            kl_loss = kl_per_dim.mean()
            loss = recon_loss + beta * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(
        {"_vae_config": {"input_dim": VAE_INPUT_DIM, "latent_dim": VAE_LATENT_DIM}, "state_dict": model.state_dict()},
        save_path,
    )
    return save_path


def train_vae_contrastive(unlabelled_embs, labelled_embs, labelled_labels, save_path,
                          pretrain_epochs=300, finetune_epochs=200, finetune_lr=5e-4,
                          contrastive_temp=0.1, seed=42):
    set_seed(seed)
    model = VAE(VAE_INPUT_DIM, VAE_LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_LR, weight_decay=1e-4)

    data = torch.from_numpy(unlabelled_embs)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=min(VAE_BATCH_SIZE, len(data)),
        shuffle=True,
    )

    model.train()
    for epoch in range(1, pretrain_epochs + 1):
        beta = min(1.0, epoch / VAE_KL_WARMUP_EPOCHS)
        for (batch_x,) in loader:
            noisy = batch_x + torch.randn_like(batch_x) * VAE_NOISE_STD
            recon, mu, logvar = model(noisy)
            recon_loss = F.mse_loss(recon, batch_x)
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_per_dim = torch.clamp(kl_per_dim, min=VAE_KL_FREE_BITS)
            kl_loss = kl_per_dim.mean()
            loss = recon_loss + beta * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    labelled_data = torch.from_numpy(labelled_embs).float()
    labelled_labels_t = torch.from_numpy(labelled_labels).long()
    ft_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(labelled_data, labelled_labels_t),
        batch_size=min(VAE_BATCH_SIZE, len(labelled_data)),
        shuffle=True,
    )

    optimizer_ft = torch.optim.Adam(model.parameters(), lr=finetune_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=finetune_epochs)

    model.train()
    for epoch in range(1, finetune_epochs + 1):
        for batch_x, batch_labels in ft_loader:
            noisy = batch_x + torch.randn_like(batch_x) * VAE_NOISE_STD
            _, mu, logvar = model(noisy)
            z = mu / (mu.norm(dim=1, keepdim=True) + 1e-8)
            loss_con = supervised_contrastive_loss(z, batch_labels, temperature=contrastive_temp)
            recon_loss = F.mse_loss(model.decode(mu), batch_x)
            loss = 0.8 * recon_loss + 0.2 * loss_con
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()
        scheduler.step()

    torch.save(
        {"_vae_config": {"input_dim": VAE_INPUT_DIM, "latent_dim": VAE_LATENT_DIM}, "state_dict": model.state_dict()},
        save_path,
    )
    return save_path


def train_vae_prototypical(unlabelled_embs, labelled_embs, labelled_labels, save_path,
                           pretrain_epochs=300, finetune_epochs=200, finetune_lr=5e-4,
                           proto_temp=0.1, seed=42):
    set_seed(seed)
    model = VAE(VAE_INPUT_DIM, VAE_LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_LR, weight_decay=1e-4)

    data = torch.from_numpy(unlabelled_embs)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=min(VAE_BATCH_SIZE, len(data)),
        shuffle=True,
    )

    model.train()
    for epoch in range(1, pretrain_epochs + 1):
        beta = min(1.0, epoch / VAE_KL_WARMUP_EPOCHS)
        for (batch_x,) in loader:
            noisy = batch_x + torch.randn_like(batch_x) * VAE_NOISE_STD
            recon, mu, logvar = model(noisy)
            recon_loss = F.mse_loss(recon, batch_x)
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_per_dim = torch.clamp(kl_per_dim, min=VAE_KL_FREE_BITS)
            kl_loss = kl_per_dim.mean()
            loss = recon_loss + beta * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    labelled_data = torch.from_numpy(labelled_embs).float()
    labelled_labels_t = torch.from_numpy(labelled_labels).long()
    ft_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(labelled_data, labelled_labels_t),
        batch_size=min(VAE_BATCH_SIZE, len(labelled_data)),
        shuffle=True,
    )

    optimizer_ft = torch.optim.Adam(model.parameters(), lr=finetune_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=finetune_epochs)

    n_speakers = len(np.unique(labelled_labels))
    model.train()
    for epoch in range(1, finetune_epochs + 1):
        for batch_x, batch_labels in ft_loader:
            noisy = batch_x + torch.randn_like(batch_x) * VAE_NOISE_STD
            _, mu, logvar = model(noisy)
            z = mu / (mu.norm(dim=1, keepdim=True) + 1e-8)
            loss_proto = prototypical_loss(z, batch_labels, n_speakers, temperature=proto_temp)
            recon_loss = F.mse_loss(model.decode(mu), batch_x)
            loss = 0.8 * recon_loss + 0.2 * loss_proto
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()
        scheduler.step()

    torch.save(
        {"_vae_config": {"input_dim": VAE_INPUT_DIM, "latent_dim": VAE_LATENT_DIM}, "state_dict": model.state_dict()},
        save_path,
    )
    return save_path


def run_ablation():
    tts = KittenTTS("KittenML/kitten-tts-nano-0.8")
    embedder = SpeakerEmbedder(EMB_MODEL_PATH)
    segmenter = SpeechSegmenter("models/model.onnx")

    print("=== Generating data ===")
    ref_paths = generate_reference_samples(tts)
    ref_embeddings = build_reference_embeddings(ref_paths, embedder)
    unlabelled_paths = generate_unlabelled_samples(tts)
    segment_embs = collect_segment_embeddings(unlabelled_paths, embedder, segmenter)
    clean_embs = collect_clean_embeddings(ref_paths, embedder)
    unlabelled_embs = np.concatenate([segment_embs, clean_embs], axis=0)
    labelled_embs, labelled_labels = collect_labelled_embeddings(ref_paths, embedder)
    turn_paths = generate_dialogue_turns(tts)

    print(f"\nTraining data: {len(unlabelled_embs)} unlabelled, {len(labelled_embs)} labelled")
    print(f"Speakers: {len(np.unique(labelled_labels))}")

    results = []

    variants = [
        ("1. Unsupervised VAE", "unsupervised", {}),
        ("2. VAE + Contrastive FT", "contrastive", {"finetune_epochs": 200}),
        ("3. VAE + Prototypical FT", "prototypical", {"finetune_epochs": 200}),
    ]

    for name, variant, kwargs in variants:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

        save_path = f"models/ablation_{variant}.pt"
        if os.path.exists(save_path):
            os.remove(save_path)

        start_time = time.time()
        if variant == "unsupervised":
            train_vae_unsupervised(unlabelled_embs, save_path, **kwargs)
        elif variant == "contrastive":
            train_vae_contrastive(unlabelled_embs, labelled_embs, labelled_labels, save_path, **kwargs)
        elif variant == "prototypical":
            train_vae_prototypical(unlabelled_embs, labelled_embs, labelled_labels, save_path, **kwargs)
        train_time = time.time() - start_time

        vae = SpeakerVAE(save_path)
        correct, total = classify_turns(turn_paths, embedder, ref_embeddings, vae=vae)
        accuracy = correct / total

        results.append((name, accuracy, train_time))
        print(f"  Accuracy: {correct}/{total} ({accuracy * 100:.1f}%)  Time: {train_time:.1f}s")

    print(f"\n{'=' * 60}")
    print("  ABLATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  {'Variant':<30s} {'Accuracy':>10s} {'Time':>8s}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 8}")
    for name, accuracy, train_time in results:
        print(f"  {name:<30s} {accuracy * 100:>9.1f}% {train_time:>7.1f}s")

    best = max(results, key=lambda x: x[1])
    print(f"\n  Best: {best[0]} ({best[1] * 100:.1f}%)")


if __name__ == "__main__":
    run_ablation()
