"""Diarization followed by speaker identification with VAE.

Combines a full dialogue into a single audio file, runs diarization to detect
speaker turns, then identifies each segment against reference speakers using a
VAE trained on unlabelled samples.

Evaluation uses a probabilistic ground truth: for each diarization segment,
overlap percentages against each known speaker's time ranges determine the
expected speaker label.

Requires the VAE dependency group:
    uv run --group vae python examples/diarization_speaker_id.py [--stitch [THRESHOLD]]

Architecture note — why standard VAE loss with no supervised terms:

The VAE uses only reconstruction (MSE) + KL divergence loss. Supervised loss
terms were tested extensively and all performed worse at this data scale
(10 training samples):

  - Cross-entropy classification: overfits, classifier dominates latent space
  - Center loss (Wen et al. 2016): noisy centroids from 3-4 samples collapse space
  - Supervised contrastive (SupCon): far more negatives than positives, collapses
  - Mixup augmentation: interpolates between speakers, creates ambiguity

With only 3-4 embeddings per speaker, any geometric constraint (pull toward
centers, push apart, classify) overconstrains the 128-dim latent space. The
standard VAE loss gives the latent space freedom to organize itself for cosine
matching. The discriminative signal comes from the training data diversity
(different phrases from the same voices) rather than from the loss function.

The classify() head and num_speakers config remain in speech_segmentation/vae.py
for future use with larger datasets (100+ samples per speaker) where supervised
losses would have enough data to generalize.
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from kittentts import KittenTTS

from speech_segmentation import Diarizer, SpeakerEmbedder, SpeechSegmenter
from speech_segmentation.augmentation import speed_perturb
from speech_segmentation.vae import VAE, SpeakerVAE, supervised_contrastive_loss

EMB_MODEL_PATH = "models/ecapa_tdnn.onnx"
SEG_MODEL_PATH = "models/model.onnx"
VAE_MODEL_PATH = "models/unsupervised_vae.pt"
TTS_SR = 24000
SILENCE_GAP_SEC = 0.5

VAE_INPUT_DIM = 192
VAE_LATENT_DIM = 64
VAE_NOISE_STD = 0.05
VAE_EPOCHS = 500
VAE_BATCH_SIZE = 32
VAE_LR = 1e-3
VAE_KL_WARMUP_EPOCHS = 100
VAE_KL_FREE_BITS = 2.0

STITCH_THRESHOLD = 0.25

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
    "Time flies when you are having fun but drags when waiting.",
    "The world is a book and those who do not travel read only one page.",
    "In the middle of difficulty lies opportunity waiting to be found.",
    "Life is short and the world is wide so explore it fully.",
    "The only limit to our realization is our own imagination.",
    "Success usually comes to those who are too busy to look for it.",
    "Do not wait for the perfect moment take the moment and make it perfect.",
    "The best way to predict the future is to create it yourself.",
    "Hard work beats talent when talent does not work hard.",
    "The journey of a thousand miles begins with a single step forward.",
    "Do what you can with what you have where you are.",
    "Everything you can imagine is real if you believe in it.",
    "The only way to do great work is to love what you do deeply.",
    "Life is what happens when you are busy making other plans today.",
    "Innovation distinguishes between a leader and a true follower.",
    "Stay hungry stay foolish and keep pushing boundaries.",
    "The future belongs to those who believe in their dreams.",
    "Every moment is a fresh beginning if you choose it to be.",
    "Do not let yesterday take up too much of today.",
    "You miss one hundred percent of the shots you never take.",
    "It always seems impossible until it is actually done.",
    "The best time to plant a tree was twenty years ago but now is second best.",
    "Success is not final and failure is not fatal it is the courage to continue.",
    "Believe you can and you are halfway there already.",
    "The only impossible journey is the one you never begin.",
    "What you get by achieving your goals is not as important as what you become.",
    "Life is either a daring adventure or nothing at all worth living.",
    "The purpose of our lives is to be happy and spread joy.",
    "Turn your wounds into wisdom and your challenges into opportunities.",
    "The mind is everything what you think you become in life.",
    "Happiness is not something ready made it comes from your own actions.",
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


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_vae(
    embeddings: np.ndarray,
    save_path: str,
    labelled_embeddings: np.ndarray | None = None,
    labelled_labels: np.ndarray | None = None,
    fine_tune_epochs: int = 200,
    fine_tune_lr: float = 5e-4,
    contrastive_temperature: float = 0.07,
    seed: int = 42,
    early_stop_patience: int = 50,
    max_grad_norm: float = 1.0,
) -> None:
    """Train VAE with optional supervised fine-tuning.

    Args:
        embeddings: Unlabelled embeddings for pre-training.
        save_path: Path to save trained model.
        labelled_embeddings: Optional labelled embeddings for fine-tuning.
        labelled_labels: Speaker labels for fine-tuning (0-indexed integers).
        fine_tune_epochs: Epochs for fine-tuning stage.
        fine_tune_lr: Learning rate for fine-tuning.
        contrastive_temperature: Temperature for contrastive loss.
        seed: Random seed for reproducibility.
        early_stop_patience: Patience for early stopping (0 = disabled).
        max_grad_norm: Maximum gradient norm for clipping (0 = disabled).
    """
    if os.path.exists(save_path):
        print(f"  Reusing existing VAE model {save_path}")
        return

    set_seed(seed)
    model = VAE(VAE_INPUT_DIM, VAE_LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_LR, weight_decay=1e-4)

    data = torch.from_numpy(embeddings)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=min(VAE_BATCH_SIZE, len(data)),
        shuffle=True,
    )

    print(f"  Stage 1 - Pre-training VAE: {VAE_INPUT_DIM} -> {VAE_LATENT_DIM} (latent), {VAE_EPOCHS} epochs")
    model.train()
    best_loss = float("inf")
    patience_counter = 0
    for epoch in range(1, VAE_EPOCHS + 1):
        beta = min(1.0, epoch / VAE_KL_WARMUP_EPOCHS)
        epoch_loss = 0.0
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
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        if epoch % 50 == 0 or epoch == 1:
            print(f"    Epoch {epoch:4d}/{VAE_EPOCHS}  loss={avg_loss:.6f}  beta={beta:.3f}")
        if early_stop_patience > 0:
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"    Early stopping at epoch {epoch} (patience={early_stop_patience})")
                    break

    if labelled_embeddings is not None and labelled_labels is not None and fine_tune_epochs > 0:
        print(f"\n  Stage 2 - Fine-tuning with contrastive loss, {fine_tune_epochs} epochs")
        labelled_data = torch.from_numpy(labelled_embeddings).float()
        labelled_labels_t = torch.from_numpy(labelled_labels).long()
        ft_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(labelled_data, labelled_labels_t),
            batch_size=min(VAE_BATCH_SIZE, len(labelled_data)),
            shuffle=True,
        )

        optimizer_ft = torch.optim.Adam(model.parameters(), lr=fine_tune_lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=fine_tune_epochs)

        model.train()
        best_ft_loss = float("inf")
        ft_patience_counter = 0
        for epoch in range(1, fine_tune_epochs + 1):
            epoch_loss = 0.0
            for batch_x, batch_labels in ft_loader:
                noisy = batch_x + torch.randn_like(batch_x) * VAE_NOISE_STD
                _, mu, logvar = model(noisy)
                z = mu / (mu.norm(dim=1, keepdim=True) + 1e-8)
                loss_con = supervised_contrastive_loss(z, batch_labels, temperature=contrastive_temperature)
                recon_loss = F.mse_loss(model.decode(mu), batch_x)
                loss = 0.8 * recon_loss + 0.2 * loss_con
                optimizer_ft.zero_grad()
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer_ft.step()
                epoch_loss += loss.item()
            scheduler.step()
            avg_ft_loss = epoch_loss / len(ft_loader)
            if epoch % 50 == 0 or epoch == 1:
                print(f"    FT Epoch {epoch:4d}/{fine_tune_epochs}  loss={avg_ft_loss:.6f}")
            if early_stop_patience > 0:
                if avg_ft_loss < best_ft_loss - 1e-6:
                    best_ft_loss = avg_ft_loss
                    ft_patience_counter = 0
                else:
                    ft_patience_counter += 1
                    if ft_patience_counter >= early_stop_patience:
                        print(f"    FT early stopping at epoch {epoch} (patience={early_stop_patience})")
                        break

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {"_vae_config": {"input_dim": VAE_INPUT_DIM, "latent_dim": VAE_LATENT_DIM}, "state_dict": model.state_dict()},
        save_path,
    )
    print(f"  Saved VAE weights to {save_path}")


def build_reference_embeddings(ref_dir, embedder):
    voices = ["Bella", "Bruno", "Luna", "Hugo", "Rosie", "Leo", "Jasper", "Kiki"]
    ref_paths = {v: [os.path.join(ref_dir, f"{v.lower()}_ref_{j}.wav") for j in range(5)] for v in voices}
    ref_embeddings = {}
    for voice, paths in ref_paths.items():
        voice_embs = []
        for path in paths:
            if not os.path.exists(path):
                continue
            audio, _ = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            emb = embedder.embed(audio)
            voice_embs.append(emb)
            print(f"    {voice} {os.path.basename(path)}: emb norm={np.linalg.norm(emb):.4f}")
        if voice_embs:
            avg_emb = np.mean(voice_embs, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            ref_embeddings[voice] = avg_emb
            print(f"  {voice}: averaged {len(voice_embs)} embeddings -> prototype norm={np.linalg.norm(avg_emb):.4f}")
    return ref_embeddings


def collect_labelled_embeddings(ref_dir, embedder):
    """Collect labelled embeddings for fine-tuning from reference files."""
    voices = ["Bella", "Bruno", "Luna", "Hugo", "Rosie", "Leo", "Jasper", "Kiki"]
    embeddings = []
    labels = []
    for label_idx, voice in enumerate(voices):
        for j in range(5):
            path = os.path.join(ref_dir, f"{voice.lower()}_ref_{j}.wav")
            if not os.path.exists(path):
                continue
            audio, _ = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            emb = embedder.embed(audio)
            embeddings.append(emb)
            labels.append(label_idx)
    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    print(f"  Collected {len(embeddings)} labelled embeddings from {len(voices)} speakers")
    return embeddings, labels


UNLABELLED_VOICES = ["Bella", "Bruno", "Luna", "Hugo", "Rosie", "Leo", "Jasper", "Kiki"]


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

        for factor in [0.9, 1.1]:
            aug_path = os.path.join(unlabelled_dir, f"unlabelled_{i:02d}_speed{factor:.1f}.wav")
            if not os.path.exists(aug_path):
                aug_audio = speed_perturb(audio, TTS_SR, factor)
                sf.write(aug_path, aug_audio, TTS_SR)
            unlabelled_paths.append(aug_path)

    print(f"  Total unlabelled files: {len(unlabelled_paths)} (with speed augmentation)")
    return unlabelled_paths


def collect_segment_embeddings(unlabelled_paths, embedder, segmenter):
    """Concatenate all available audio, segment with pyannote, embed each segment.

    Uses unlabelled phrases + speaker recordings + dialogue turns to build
    a large, diverse training set that matches the diarization query domain.
    """
    FRAME_STEP = 270
    MIN_EMBED_SAMPLES = 12800

    extra_audio_files = [
        "bella.wav", "bruno.wav", "luna.wav",
        "bella_contrastive.wav", "bruno_contrastive.wav",
    ]
    turn_paths = sorted(
        os.path.join("turns", f) for f in os.listdir("turns") if f.endswith(".wav")
    ) if os.path.isdir("turns") else []

    all_paths = list(unlabelled_paths) + extra_audio_files + turn_paths
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
    print(f"  Combined {sum(1 for p in all_paths if os.path.exists(p))} audio files ({len(combined_tts) / TTS_SR:.1f}s total)")

    combined_16k = np.interp(
        np.linspace(0, len(combined_tts) - 1, int(len(combined_tts) * 16000 / TTS_SR)),
        np.arange(len(combined_tts)),
        combined_tts,
    ).astype(np.float32)

    segments = segmenter.segment(combined_16k)
    long_segs = [s for s in segments if (s.end_frame - s.start_frame) * FRAME_STEP >= MIN_EMBED_SAMPLES]
    print(f"  Segmenter found {len(segments)} segments, {len(long_segs)} above {MIN_EMBED_SAMPLES / 16000:.1f}s threshold")

    embeddings = []
    for seg in long_segs:
        start_sample = seg.start_frame * FRAME_STEP
        end_sample = seg.end_frame * FRAME_STEP
        seg_audio = combined_16k[start_sample:end_sample]
        emb = embedder.embed(seg_audio)
        embeddings.append(emb)

    arr = np.array(embeddings, dtype=np.float32)
    print(f"  Collected {len(arr)} segment embeddings for VAE training")
    return arr


def collect_clean_embeddings(ref_dir, embedder):
    """Embed clean reference files so VAE learns both segment and clean distributions."""
    voices = ["Bella", "Bruno", "Luna"]
    embeddings = []
    for v in voices:
        for j in range(3):
            path = os.path.join(ref_dir, f"{v.lower()}_ref_{j}.wav")
            audio, _ = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            emb = embedder.embed(audio)
            embeddings.append(emb)
    arr = np.array(embeddings, dtype=np.float32)
    print(f"  Collected {len(arr)} clean reference embeddings for VAE training")
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


def combine_turns(turn_paths, output_path, silence_gap_sec=SILENCE_GAP_SEC, tts_sr=TTS_SR):
    """Concatenate turn WAVs into a single audio file with silence gaps.

    Returns:
        Tuple of (combined audio array at 16kHz, ground truth ranges).
        ground truth ranges is a list of (speaker_name, start_sec, end_sec).
    """
    silence_samples = int(silence_gap_sec * tts_sr)
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
            cursor_sec += silence_gap_sec

    combined_tts = np.concatenate(chunks)
    sf.write(output_path, combined_tts, tts_sr)
    print(f"  Saved combined audio: {output_path} ({len(combined_tts) / tts_sr:.2f}s)")

    combined_16k = np.interp(
        np.linspace(0, len(combined_tts) - 1, int(len(combined_tts) * 16000 / tts_sr)),
        np.arange(len(combined_tts)),
        combined_tts,
    ).astype(np.float32)

    return combined_16k, ground_truth


def overlap_seconds(seg_start, seg_end, gt_start, gt_end):
    """Compute overlap in seconds between a segment and a ground truth range."""
    return max(0.0, min(seg_end, gt_end) - max(seg_start, gt_start))


def evaluate_diarization(matches, ground_truth):
    """Evaluate diarization segments against probabilistic ground truth.

    For each segment, compute overlap % with each speaker's ground truth ranges.
    The speaker with highest overlap is the expected label.
    Shows similarity scores for each segment to explain predictions.
    """
    all_speakers = sorted(set(s for s, _, _ in ground_truth))

    print(f"\n  {'Seg#':>4s}  {'Time Range':>18s}  {'Expected':>10s}  {'Predicted':>10s}  {'Sim':>5s}  {'OK?':>4s}  Similarities")
    print(f"  {'-' * 4}  {'-' * 18}  {'-' * 10}  {'-' * 10}  {'-' * 5}  {'-' * 4}  {'-' * 40}")

    correct = 0
    for i, m in enumerate(matches):
        seg_duration = m.end_time - m.start_time
        if seg_duration <= 0:
            continue

        overlaps = {}
        for speaker in all_speakers:
            total = sum(
                overlap_seconds(m.start_time, m.end_time, gt_start, gt_end)
                for s, gt_start, gt_end in ground_truth
                if s == speaker
            )
            overlaps[speaker] = total

        expected = max(overlaps, key=overlaps.get)
        is_correct = m.speaker == expected
        if is_correct:
            correct += 1

        mark = "" if is_correct else " [WRONG]"
        sim_str = " ".join(f"{n}={s:.3f}" for n, s in m.all_sims.items())
        time_range = f"{m.start_time:6.2f}s-{m.end_time:6.2f}s"
        print(
            f"  {i + 1:4d}  {time_range:>18s}  {expected:>10s}  {m.speaker:>10s}  {m.similarity:5.3f}  {'OK' if is_correct else '  ':>4s}  [{sim_str}]{mark}"
        )

    total = len(matches)
    print(f"\n  Accuracy: {correct}/{total} segments correct ({correct / total * 100:.1f}%)")
    return correct, total


def main():
    parser = argparse.ArgumentParser(description="Diarization with speaker identification (unsupervised VAE)")
    parser.add_argument(
        "--stitch",
        nargs="?",
        const=STITCH_THRESHOLD,
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Merge consecutive same-speaker segments when raw ECAPA-TDNN cosine similarity exceeds threshold "
        f"(default: {STITCH_THRESHOLD})",
    )
    parser.add_argument(
        "--stitch-raw",
        action="store_true",
        help="Use raw ECAPA-TDNN embeddings for speaker label determination during stitching",
    )
    args = parser.parse_args()

    tts = KittenTTS("KittenML/kitten-tts-nano-0.8")
    print("Loading models...")
    embedder = SpeakerEmbedder(EMB_MODEL_PATH)
    segmenter = SpeechSegmenter(SEG_MODEL_PATH)

    print("\n=== Step 1: Building few-shot reference embeddings (ECAPA-TDNN) ===")
    ref_embeddings = build_reference_embeddings("refs", embedder)

    print("\n=== Step 2: Generating unlabelled samples for VAE training ===")
    unlabelled_paths = generate_unlabelled_samples(tts, "unlabelled")

    print("\n=== Step 3: Collecting embeddings for VAE training ===")
    segment_embs = collect_segment_embeddings(unlabelled_paths, embedder, segmenter)
    clean_embs = collect_clean_embeddings("refs", embedder)
    unlabelled_embs = np.concatenate([segment_embs, clean_embs], axis=0)
    print(f"  Combined: {len(segment_embs)} segment + {len(clean_embs)} clean = {len(unlabelled_embs)} total")

    print("\n=== Step 3b: Applying embedding augmentation ===")
    print(f"  Training embeddings: {len(unlabelled_embs)} (speed-augmented at audio level, no embedding noise)")

    print("\n=== Step 3c: Collecting labelled embeddings for fine-tuning ===")
    labelled_embs, labelled_labels = collect_labelled_embeddings("refs", embedder)

    print("\n=== Step 4: Training VAE with pre-training + fine-tuning ===")
    train_vae(
        unlabelled_embs,
        VAE_MODEL_PATH,
        labelled_embeddings=labelled_embs,
        labelled_labels=labelled_labels,
        fine_tune_epochs=0,
        fine_tune_lr=5e-4,
        contrastive_temperature=0.1,
    )

    print("\n=== Step 5: Loading trained VAE ===")
    vae = SpeakerVAE(VAE_MODEL_PATH)
    print(f"  VAE: {vae.INPUT_DIM} -> {vae.LATENT_DIM} (latent)")

    print("\n=== Step 6: Generating dialogue turns ===")
    turn_paths = generate_dialogue_turns(tts, "turns")

    print("\n=== Step 7: Combining turns into single audio file ===")
    combined_path = "combined_dialogue.wav"
    combined_audio, ground_truth = combine_turns(turn_paths, combined_path)

    print("\n=== Step 8: Running diarization on combined audio ===")
    diarizer = Diarizer(segmenter, embedder, vae=vae)
    diarizer.build_references(ref_embeddings)
    stitch_kw = {}
    if args.stitch is not None:
        stitch_kw["stitch_threshold"] = args.stitch
    if args.stitch_raw:
        stitch_kw["stitch_raw"] = True
    matches, raw_segments = diarizer.diarize(combined_audio, **stitch_kw)
    print(f"  Found {len(matches)} segments from {len(raw_segments)} raw segments")

    print("\n=== Step 9: Evaluating against probabilistic ground truth ===")
    evaluate_diarization(matches, ground_truth)


if __name__ == "__main__":
    main()
