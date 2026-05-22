import onnx
import urllib.request
from onnx import shape_inference
from pathlib import Path

model_id = "onnx-community/pyannote-segmentation-3.0"

model_path = "model.onnx"

if not Path(model_path).exists():
    urllib.request.urlretrieve(
        f"https://huggingface.co/{model_id}/resolve/main/onnx/model.onnx",
        model_path,
    )

model = onnx.load("model.onnx")

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
    onnx.save(model, "model_with_embedding.onnx")
    print(f"Added '{embedding_name}' as embedding output")
