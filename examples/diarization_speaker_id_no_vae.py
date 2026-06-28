"""Diarization followed by speaker identification (no VAE).

Combines a full dialogue into a single audio file, runs diarization to detect
speaker turns, then identifies each segment against reference speakers using
raw ECAPA-TDNN embeddings without any VAE projection.

Evaluation uses a probabilistic ground truth: for each diarization segment,
overlap percentages against each known speaker's time ranges determine the
expected speaker label.

Usage:
    uv run python examples/diarization_speaker_id_no_vae.py [--stitch [THRESHOLD]]
"""

import argparse
import os

import numpy as np
import soundfile as sf
from kittentts import KittenTTS

from speech_segmentation import Diarizer, SpeakerEmbedder, SpeechSegmenter

EMB_MODEL_PATH = "models/ecapa_tdnn.onnx"
NORM_MEAN_PATH = "models/ecapa_norm_mean.npy"
SEG_MODEL_PATH = "models/model.onnx"
TTS_SR = 24000
SILENCE_GAP_SEC = 0.5

STITCH_THRESHOLD = 0.25

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
    parser = argparse.ArgumentParser(description="Diarization with speaker identification (no VAE)")
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
    embedder = SpeakerEmbedder(EMB_MODEL_PATH, NORM_MEAN_PATH)
    segmenter = SpeechSegmenter(SEG_MODEL_PATH)

    print("\n=== Step 1: Building few-shot reference embeddings (ECAPA-TDNN) ===")
    ref_embeddings = build_reference_embeddings("refs", embedder)

    print("\n=== Step 2: Generating dialogue turns ===")
    turn_paths = generate_dialogue_turns(tts, "turns")

    print("\n=== Step 3: Combining turns into single audio file ===")
    combined_path = "combined_dialogue.wav"
    combined_audio, ground_truth = combine_turns(turn_paths, combined_path)

    print("\n=== Step 4: Running diarization on combined audio (no VAE) ===")
    diarizer = Diarizer(segmenter, embedder)
    diarizer.build_references(ref_embeddings)
    stitch_kw = {}
    if args.stitch is not None:
        stitch_kw["stitch_threshold"] = args.stitch
    if args.stitch_raw:
        stitch_kw["stitch_raw"] = True
    matches, raw_segments = diarizer.diarize(combined_audio, **stitch_kw)
    print(f"  Found {len(matches)} segments from {len(raw_segments)} raw segments")

    print("\n=== Step 5: Evaluating against probabilistic ground truth ===")
    evaluate_diarization(matches, ground_truth)


if __name__ == "__main__":
    main()
