"""
Export pyannote-segmentation-3.0 with embedding output.

Patches the ONNX graph to expose the last LeakyRelu layer (128-dim)
as a second output alongside the existing logits. This gives access
to the model's penultimate layer — a per-frame embedding that captures
speech characteristics before the final classifier.

Usage (from project root):
    uv run --group export python scripts/pyannote_embedding_export.py

Outputs (in models/):
    model_with_embedding.onnx — 2 outputs: [logits, embedding]
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    src_path = MODELS_DIR / "model.onnx"
    dst_path = MODELS_DIR / "model_with_embedding.onnx"

    print(f"Loading {src_path}...")
    model = onnx.load(str(src_path))

    print("Running shape inference...")
    inferred = shape_inference.infer_shapes(model)

    leaky_relu_nodes = [n for n in inferred.graph.node if n.op_type == "LeakyRelu"]
    print(f"  Found {len(leaky_relu_nodes)} LeakyRelu nodes")

    embedding_node = leaky_relu_nodes[-1]
    embedding_name = embedding_node.output[0]

    embedding_info = next(
        (v for v in inferred.graph.value_info if v.name == embedding_name), None
    )
    if embedding_info is None:
        raise ValueError(f"Could not find shape info for '{embedding_name}'")

    print(f"  Embedding node: {embedding_name}")
    model.graph.output.append(embedding_info)

    onnx.checker.check_model(model)

    print(f"Saving {dst_path}...")
    onnx.save(model, str(dst_path))

    print("Verifying with onnxruntime...")
    session = ort.InferenceSession(str(dst_path))
    outputs = session.get_outputs()
    print(f"  Outputs: {[o.name for o in outputs]}")

    dummy = np.random.randn(1, 1, 16000).astype(np.float32)
    logits, embedding = session.run(None, {"input_values": dummy})
    print(f"  logits shape:    {logits.shape}")
    print(f"  embedding shape: {embedding.shape}")

    print(f"\nDone. Saved {dst_path}")


if __name__ == "__main__":
    main()
