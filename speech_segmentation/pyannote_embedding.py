"""Speech embedding extraction using pyannote ONNX model."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

TARGET_SR = 16000


class PyannoteEmbedder:
    """Extract 128-dim embeddings from pyannote model's penultimate layer.

    Requires a patched ONNX model with embedding output (see
    scripts/pyannote_embedding_export.py).

    Args:
        model_path: Path to the patched ONNX model (model_with_embedding.onnx).
    """

    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(model_path)

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        """Extract embedding from 16kHz audio.

        Averages across all frames to produce a single 128-dim vector.

        Args:
            audio_16k: Audio samples at 16kHz, float32.

        Returns:
            L2-normalized 128-dim embedding vector.
        """
        frames = self.embed_frames(audio_16k)
        emb = frames.mean(axis=0)
        emb = emb / np.linalg.norm(emb)
        return emb

    def embed_frames(self, audio_16k: np.ndarray) -> np.ndarray:
        """Extract per-frame embeddings.

        Args:
            audio_16k: Audio samples at 16kHz, float32.

        Returns:
            Array of shape (n_frames, 128).
        """
        _, embedding = self.session.run(
            None,
            {"input_values": audio_16k[np.newaxis, np.newaxis, :].astype(np.float32)},
        )
        return embedding[0]
