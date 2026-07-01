"""
Test speaker discrimination of pyannote LSTM embeddings (256-dim).

Compares the LSTM output (before MLP head) with the penultimate layer
(128-dim) to see which is more speaker-discriminative.

Usage:
    uv run --group vae python scripts/pyannote_lstm_test.py
"""

import os
from collections import defaultdict

import numpy as np
import onnxruntime as ort

from speech_segmentation.audio import load_audio

MODELS_DIR = "models"


class Embedder:
    def __init__(self, model_path: str, output_idx: int) -> None:
        self.session = ort.InferenceSession(model_path)
        self.output_idx = output_idx

    def embed_frames(self, audio_16k: np.ndarray) -> np.ndarray:
        _, output = self.session.run(
            None,
            {"input_values": audio_16k[np.newaxis, np.newaxis, :].astype(np.float32)},
        )
        return output[0]  # (n_frames, dim)

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        frames = self.embed_frames(audio_16k)
        emb = frames.mean(axis=0)
        return emb / np.linalg.norm(emb)


def classify(proto: np.ndarray, turns: list[tuple[np.ndarray, str]], names: list[str]) -> float:
    correct = 0
    for emb, lbl in turns:
        pred = names[int(np.argmax(proto @ emb))]
        correct += int(pred == lbl)
    return correct / len(turns)


def main():
    print("=" * 70)
    print("PYANNOTE LSTM EMBEDDING SPEAKER DISCRIMINATION TEST")
    print("=" * 70)
    print()

    # Load both models
    emb_128 = Embedder(f"{MODELS_DIR}/model_with_embedding.onnx", 1)
    emb_256 = Embedder(f"{MODELS_DIR}/model_with_lstm.onnx", 1)

    # Build prototypes from refs/
    ref_embs: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"128": [], "256": []})
    for fname in sorted(os.listdir("refs")):
        if not fname.endswith(".wav"):
            continue
        speaker = fname.rsplit("_ref_", 1)[0]
        audio, _ = load_audio(os.path.join("refs", fname))
        ref_embs[speaker]["128"].append(emb_128.embed(audio))
        ref_embs[speaker]["256"].append(emb_256.embed(audio))

    names = sorted(ref_embs.keys())
    proto_128 = np.array([np.mean(ref_embs[n]["128"], axis=0) for n in names])
    proto_128 /= np.linalg.norm(proto_128, axis=1, keepdims=True)
    proto_256 = np.array([np.mean(ref_embs[n]["256"], axis=0) for n in names])
    proto_256 /= np.linalg.norm(proto_256, axis=1, keepdims=True)

    # Load test turns
    turns_128 = []
    turns_256 = []
    for fname in sorted(os.listdir("turns")):
        if not fname.endswith(".wav"):
            continue
        speaker = fname.split("_")[2].replace(".wav", "")
        audio, _ = load_audio(os.path.join("turns", fname))
        turns_128.append((emb_128.embed(audio), speaker))
        turns_256.append((emb_256.embed(audio), speaker))

    acc_128 = classify(proto_128, turns_128, names)
    acc_256 = classify(proto_256, turns_256, names)

    # Compute prototype separation
    def sep(embs):
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        n = embs / (norms + 1e-8)
        sim = n @ n.T
        np.fill_diagonal(sim, 0)
        return float(sim.sum()) / (len(embs) * (len(embs) - 1))

    print(f"{'Layer':<25} {'Dim':>6} {'Accuracy':>10} {'Proto Sep':>10}")
    print("-" * 55)
    print(f"{'Penultimate (MLP)':<25} {'128':>6} {acc_128:>9.1%} {sep(proto_128):>10.4f}")
    print(f"{'LSTM output':<25} {'256':>6} {acc_256:>9.1%} {sep(proto_256):>10.4f}")
    print(f"{'Delta':<25} {'':>6} {acc_256 - acc_128:>+9.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
