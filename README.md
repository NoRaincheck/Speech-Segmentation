# Speech Segmentation / Speaker Diarization

Offline speaker diarization using ONNX models (pyannote-segmentation-3.0).

## Models

| Model | Description |
|---|---|
| `model.onnx` | Base segmentation model from [onnx-community/pyannote-segmentation-3.0](https://huggingface.co/onnx-community/pyannote-segmentation-3.0) |
| `model_with_embedding.onnx` | Extended version with speaker embeddings as an additional output (generated via `speech_embedding_export.py`) |

## Usage

### Basic diarization

Outputs detected speakers with timestamps and confidence scores:

```bash
uv run python speech_diarizer.py
```

Automatically downloads the model and sample audio (`mlk.wav`), then prints segments like:

```
  SPEAKER_01      0.37s -    2.84s  (conf=0.951)
  SPEAKER_02      2.84s -    5.21s  (conf=0.876)
```

### Diarization with embeddings

Extract per-segment speaker embeddings alongside timestamps:

```bash
uv run python speech_embedding.py
```

Output includes embedding dimensions for each segment, useful for downstream clustering or verification.

### Export model with embeddings

Re-exports the base ONNX model to include the LeakyRelu activation (speaker embeddings) as a graph output:

```bash
uv run python speech_embedding_export.py
# Produces: model_with_embedding.onnx
```

## Setup

Dependencies are managed via `uv`:

```bash
pip install uv          # if not already installed
uv sync                 # installs dependencies from pyproject.toml
```
