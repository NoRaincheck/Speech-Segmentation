"""
Export SpeechBrain ECAPA-TDNN speaker embedding model to ONNX.

Loads the pretrained spkrec-ecapa-voxceleb model and exports the
ECAPA_TDNN embedding network to ONNX format for inference without
the full SpeechBrain dependency.

Usage (from project root):
    python conversion/ecapa_tdnn_export.py

Outputs (in models/):
    ecapa_tdnn.onnx       — ONNX model (input: [B, T, 80], output: [B, 1, 192])
    ecapa_norm_mean.npy   — Global embedding normalization mean (192,)
"""

from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ECAPAWrapper(torch.nn.Module):
    """Thin wrapper around ECAPA_TDNN for clean ONNX export."""

    def __init__(self, ecapa_model):
        super().__init__()
        self.model = ecapa_model

    def forward(self, x):
        return self.model(x, lengths=None)


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("Loading SpeechBrain ECAPA-TDNN model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )

    ecapa_model = classifier.mods["embedding_model"]
    ecapa_model.eval()

    mean_var_norm_emb = classifier.mods["mean_var_norm_emb"]

    print(f"  ECAPA_TDNN loaded: {sum(p.numel() for p in ecapa_model.parameters())} parameters")

    norm_mean = mean_var_norm_emb.glob_mean.numpy()
    norm_path = MODELS_DIR / "ecapa_norm_mean.npy"
    np.save(norm_path, norm_mean)
    print(f"  Saved {norm_path}")

    wrapper = ECAPAWrapper(ecapa_model)
    wrapper.eval()

    dummy_input = torch.randn(1, 100, 80)
    onnx_path = MODELS_DIR / "ecapa_tdnn.onnx"
    print(f"\nExporting to {onnx_path} (input shape: {list(dummy_input.shape)})...")

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        input_names=["input_values"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_values": {1: "time"},
            "embeddings": {1: "time"},
        },
        opset_version=18,
        do_constant_folding=True,
    )

    print(f"  Saved {onnx_path}")

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    onnx_out = session.run(None, {"input_values": dummy_input.numpy()})
    print(f"  ONNX output shape: {onnx_out[0].shape}")

    with torch.no_grad():
        torch_out = wrapper(dummy_input).numpy()
    print(f"  PyTorch output shape: {torch_out.shape}")

    cos_sim = np.dot(torch_out.flatten(), onnx_out[0].flatten()) / (
        np.linalg.norm(torch_out) * np.linalg.norm(onnx_out[0])
    )
    max_diff = np.max(np.abs(torch_out - onnx_out[0]))
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max absolute diff: {max_diff:.8f}")

    print(f"\nDone. Files saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
