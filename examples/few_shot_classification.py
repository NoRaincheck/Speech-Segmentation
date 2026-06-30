"""Few-shot speaker classification: identify speakers from reference audio.

This example answers "who is speaking?" for each turn in a conversation.
It needs a few reference clips per speaker (as few as 1, but 3-5 is
better) to build a speaker prototype, then classifies each turn by
cosine similarity against all prototypes.

Pipeline:
    reference audio → SpeakerEmbedder → averaged prototype per speaker
    dialogue turn   → SpeakerEmbedder → cosine similarity → best match

Usage:
    uv run python examples/few_shot_classification.py
    uv run python examples/few_shot_classification.py --refs-dir refs --turns-dir turns
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np

from speech_segmentation import SpeakerEmbedder
from speech_segmentation.audio import load_audio

# Path to the ECAPA-TDNN ONNX model.
# This model extracts 192-dimensional speaker embeddings from 16kHz audio.
# The embeddings are L2-normalized, so cosine similarity = dot product.
MODEL_PATH = "models/ecapa_tdnn.onnx"


def build_prototypes(
    refs_dir: str, embedder: SpeakerEmbedder
) -> tuple[np.ndarray, list[str]]:
    """Load reference audio per speaker and average their embeddings.

    Each speaker should have 3-5 reference clips in refs_dir, named
    like "bella_ref_0.wav", "bella_ref_1.wav", etc. The speaker name
    is extracted from the filename before "_ref_".

    Averaging multiple embeddings creates a robust "prototype" that
    captures the speaker's core characteristics while smoothing out
    phrase-specific variation.

    Returns:
        (proto_matrix, names) where proto_matrix is (n_speakers, 192)
        and names is the ordered list of speaker names.
    """
    # Group reference files by speaker name
    speaker_files: dict[str, list[str]] = defaultdict(list)
    for fname in sorted(os.listdir(refs_dir)):
        if not fname.endswith(".wav"):
            continue
        # Extract speaker name: "bella_ref_0.wav" → "bella"
        speaker = fname.rsplit("_ref_", 1)[0]
        speaker_files[speaker].append(os.path.join(refs_dir, fname))

    names = sorted(speaker_files.keys())
    prototypes = []

    for name in names:
        # Embed each reference clip
        embs = []
        for path in speaker_files[name]:
            audio, _ = load_audio(path)
            embs.append(embedder.embed(audio))

        # Average into a single prototype and re-normalize
        proto = np.mean(embs, axis=0)
        proto = proto / np.linalg.norm(proto)
        prototypes.append(proto)

    return np.array(prototypes), names


def classify_turns(
    turns_dir: str,
    embedder: SpeakerEmbedder,
    proto_matrix: np.ndarray,
    names: list[str],
) -> list[tuple[str, str, float, dict[str, float]]]:
    """Classify each dialogue turn against the prototype matrix.

    For each turn:
        1. Extract its ECAPA-TDNN embedding
        2. Compute cosine similarity (dot product) against all prototypes
        3. Assign the speaker with highest similarity

    Returns a list of (filename, true_speaker, predicted_speaker, similarity,
    all_similarities) tuples.
    """
    results = []

    for fname in sorted(os.listdir(turns_dir)):
        if not fname.endswith(".wav"):
            continue

        # Extract ground truth speaker from filename:
        # "turn_00_bella.wav" → "bella"
        true_speaker = fname.split("_")[2].replace(".wav", "")

        # Embed the turn
        audio, _ = load_audio(os.path.join(turns_dir, fname))
        emb = embedder.embed(audio)

        # Cosine similarity = dot product (embeddings are L2-normalized)
        sims = proto_matrix @ emb
        best_idx = int(np.argmax(sims))
        all_sims = {name: float(s) for name, s in zip(names, sims)}

        results.append(
            (fname, true_speaker, names[best_idx], float(sims[best_idx]), all_sims)
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Few-shot speaker classification from reference audio."
    )
    parser.add_argument(
        "--refs-dir", default="refs", help="Directory with reference audio clips"
    )
    parser.add_argument(
        "--turns-dir", default="turns", help="Directory with dialogue turns to classify"
    )
    args = parser.parse_args()

    # Step 1: Load the speaker embedding model.
    # ECAPA-TDNN is a neural network trained on VoxCeleb that maps audio
    # to a 192-dimensional vector capturing speaker identity. It processes
    # 80-dim filterbank features through a temporal convolutional network
    # with attention, producing a single embedding per audio clip.
    print("Loading speaker embedder...")
    t0 = time.perf_counter()
    embedder = SpeakerEmbedder(MODEL_PATH)
    print(f"  Loaded in {time.perf_counter() - t0:.2f}s")

    # Step 2: Build speaker prototypes from reference audio.
    # A prototype is the average of multiple embeddings from the same speaker.
    # This is more robust than using a single clip because it smooths out
    # phrase-specific and recording-specific variation.
    print(f"\nBuilding prototypes from {args.refs_dir}/...")
    t0 = time.perf_counter()
    proto_matrix, names = build_prototypes(args.refs_dir, embedder)
    n_refs = sum(
        len(f)
        for f in [
            [n for n in os.listdir(args.refs_dir) if n.startswith(s)]
            for s in names
        ]
    )
    print(f"  {len(names)} speakers, {n_refs} reference clips")
    print(f"  Built in {time.perf_counter() - t0:.2f}s")

    # Step 3: Classify each dialogue turn.
    # For each turn, we compute its embedding and find the closest prototype
    # via cosine similarity (dot product of L2-normalized vectors).
    print(f"\nClassifying turns from {args.turns_dir}/...")
    t0 = time.perf_counter()
    results = classify_turns(args.turns_dir, embedder, proto_matrix, names)
    elapsed = time.perf_counter() - t0
    print(f"  {len(results)} turns classified in {elapsed:.2f}s\n")

    # Step 4: Print results.
    print(f"{'Turn':<24} {'True':<12} {'Predicted':<12} {'Sim':>6} {'OK?':>5}")
    print("-" * 62)
    correct = 0
    for fname, true_spk, pred_spk, sim, _ in results:
        is_correct = true_spk == pred_spk
        correct += int(is_correct)
        mark = "yes" if is_correct else "MISS"
        print(f"{fname:<24} {true_spk:<12} {pred_spk:<12} {sim:>6.3f} {mark:>5}")

    accuracy = correct / len(results) if results else 0
    print(f"\nAccuracy: {correct}/{len(results)} = {accuracy:.0%}")


if __name__ == "__main__":
    main()
