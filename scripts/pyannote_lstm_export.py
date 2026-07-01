"""
Export pyannote-segmentation-3.0 with LSTM embedding output.

Adds the bidirectional LSTM output (256-dim) as a second output.
This is the richest intermediate representation — before the task-specific
MLP head compresses it to 128-dim for segmentation.

Usage (from project root):
    uv run --group export python scripts/pyannote_lstm_export.py

Outputs (in models/):
    model_with_lstm.onnx — 2 outputs: [logits, lstm_embedding]
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    src_path = MODELS_DIR / "model.onnx"
    dst_path = MODELS_DIR / "model_with_lstm.onnx"

    print(f"Loading {src_path}...")
    model = onnx.load(str(src_path))

    print("Running shape inference...")
    inferred = shape_inference.infer_shapes(model)

    # The LSTM output is Transpose_5_output_0: [batch, frames, 256]
    lstm_name = "/lstm/Transpose_5_output_0"
    lstm_info = next(
        (v for v in inferred.graph.value_info if v.name == lstm_name), None
    )
    if lstm_info is None:
        raise ValueError(f"Could not find shape info for '{lstm_name}'")

    print(f"  LSTM node: {lstm_name}")
    model.graph.output.append(lstm_info)

    onnx.checker.check_model(model)

    print(f"Saving {dst_path}...")
    onnx.save(model, str(dst_path))

    print("Verifying with onnxruntime...")
    session = ort.InferenceSession(str(dst_path))
    outputs = session.get_outputs()
    print(f"  Outputs: {[o.name for o in outputs]}")

    dummy = np.random.randn(1, 1, 16000).astype(np.float32)
    logits, lstm_emb = session.run(None, {"input_values": dummy})
    print(f"  logits shape:       {logits.shape}")
    print(f"  lstm embedding shape: {lstm_emb.shape}")

    print(f"\nDone. Saved {dst_path}")


if __name__ == "__main__":
    main()
