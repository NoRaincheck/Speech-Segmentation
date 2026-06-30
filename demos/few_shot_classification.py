"""Few-shot speaker classification demo.

Loads reference audio clips per speaker, builds averaged prototypes from
their ECAPA-TDNN embeddings, then classifies each dialogue turn via cosine
similarity against the reference matrix.

Usage:
    uv run python demos/few_shot_classification.py
    uv run python demos/few_shot_classification.py --refs-dir refs --turns-dir turns
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np

from speech_segmentation import SpeakerEmbedder
from speech_segmentation.audio import load_audio

MODELS_DIR = "models"


def build_prototypes(refs_dir: str, embedder: SpeakerEmbedder) -> tuple[np.ndarray, list[str]]:
    """Load reference audio per speaker and average their embeddings."""
    speaker_files: dict[str, list[str]] = defaultdict(list)
    for fname in sorted(os.listdir(refs_dir)):
        if not fname.endswith(".wav"):
            continue
        speaker = fname.rsplit("_ref_", 1)[0]
        speaker_files[speaker].append(os.path.join(refs_dir, fname))

    names = sorted(speaker_files.keys())
    prototypes = []
    for name in names:
        embs = []
        for path in speaker_files[name]:
            audio, _ = load_audio(path)
            embs.append(embedder.embed(audio))
        prototypes.append(np.mean(embs, axis=0))

    proto_matrix = np.array(prototypes)
    norms = np.linalg.norm(proto_matrix, axis=1, keepdims=True)
    proto_matrix = proto_matrix / norms
    return proto_matrix, names


def classify_turns(
    turns_dir: str, embedder: SpeakerEmbedder, proto_matrix: np.ndarray, names: list[str]
) -> list[tuple[str, str, float, dict[str, float]]]:
    """Classify each turn file against the prototype matrix."""
    results = []
    for fname in sorted(os.listdir(turns_dir)):
        if not fname.endswith(".wav"):
            continue
        true_speaker = fname.split("_")[2].replace(".wav", "")
        audio, _ = load_audio(os.path.join(turns_dir, fname))
        emb = embedder.embed(audio)
        sims = proto_matrix @ emb
        best_idx = int(np.argmax(sims))
        all_sims = {name: float(s) for name, s in zip(names, sims)}
        results.append((true_speaker, names[best_idx], float(sims[best_idx]), all_sims))
    return results


def main():
    parser = argparse.ArgumentParser(description="Few-shot speaker classification demo")
    parser.add_argument("--refs-dir", default="refs", help="Directory with reference audio")
    parser.add_argument("--turns-dir", default="turns", help="Directory with dialogue turns")
    args = parser.parse_args()

    print("Loading speaker embedder...")
    t0 = time.perf_counter()
    embedder = SpeakerEmbedder(f"{MODELS_DIR}/ecapa_tdnn.onnx")
    print(f"  Loaded in {time.perf_counter() - t0:.2f}s")

    print(f"\nBuilding prototypes from {args.refs_dir}/...")
    t0 = time.perf_counter()
    proto_matrix, names = build_prototypes(args.refs_dir, embedder)
    print(f"  {len(names)} speakers, {len(os.listdir(args.refs_dir))} reference clips")
    print(f"  Built in {time.perf_counter() - t0:.2f}s")

    print(f"\nClassifying turns from {args.turns_dir}/...")
    t0 = time.perf_counter()
    results = classify_turns(args.turns_dir, embedder, proto_matrix, names)
    elapsed = time.perf_counter() - t0
    print(f"  {len(results)} turns classified in {elapsed:.2f}s\n")

    print(f"{'Turn':<20} {'True':<12} {'Predicted':<12} {'Sim':>6} {'Correct':>8}")
    print("-" * 62)
    correct = 0
    for fname, (true_spk, pred_spk, sim, _) in zip(
        sorted(f for f in os.listdir(args.turns_dir) if f.endswith(".wav")), results
    ):
        is_correct = true_spk == pred_spk
        correct += int(is_correct)
        mark = "  OK" if is_correct else "  MISS"
        print(f"{fname:<20} {true_spk:<12} {pred_spk:<12} {sim:>6.3f}{mark}")

    accuracy = correct / len(results) if results else 0
    print(f"\nAccuracy: {correct}/{len(results)} = {accuracy:.1%}")


if __name__ == "__main__":
    main()
