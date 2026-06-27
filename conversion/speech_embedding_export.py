"""
Export pyannote-segmentation-3.0 with embedding output.

Adds the last LeakyRelu node as an additional graph output so the model
returns both classification logits and speaker embeddings.

Usage (from project root):
    python conversion/speech_embedding_export.py

Outputs (in models/):
    model_with_embedding.onnx
"""

import urllib.request
from pathlib import Path

import onnx
from onnx import shape_inference

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_ID = "onnx-community/pyannote-segmentation-3.0"


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "model.onnx"

    if not model_path.exists():
        urllib.request.urlretrieve(
            f"https://huggingface.co/{MODEL_ID}/resolve/main/onnx/model.onnx",
            model_path,
        )
        print(f"Downloaded {model_path}")

    model = onnx.load(str(model_path))

    existing_outputs = {o.name for o in model.graph.output}

    inferred = shape_inference.infer_shapes(model)

    leaky_relu_nodes = [n for n in inferred.graph.node if n.op_type == "LeakyRelu"]
    if not leaky_relu_nodes:
        raise ValueError("No LeakyRelu nodes found")

    embedding_node = leaky_relu_nodes[-1]
    embedding_name = embedding_node.output[0]

    if embedding_name in existing_outputs:
        print(f"'{embedding_name}' is already a model output")
    else:
        embedding_info = next(
            (v for v in inferred.graph.value_info if v.name == embedding_name),
            None,
        )
        if embedding_info is None:
            raise ValueError(f"Could not find shape info for '{embedding_name}'")

        model.graph.output.append(embedding_info)

        onnx.checker.check_model(model)
        out_path = MODELS_DIR / "model_with_embedding.onnx"
        onnx.save(model, str(out_path))
        print(f"Added '{embedding_name}' as embedding output -> {out_path}")


if __name__ == "__main__":
    main()
