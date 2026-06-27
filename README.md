# Speech Segmentation / Speaker Diarization

Offline speaker diarization using ONNX models (pyannote-segmentation-3.0 + ECAPA-TDNN).

## Models

| Model                              | Description                                                                                                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `models/model.onnx`                | Base segmentation model from [onnx-community/pyannote-segmentation-3.0](https://huggingface.co/onnx-community/pyannote-segmentation-3.0) |
| `models/model_with_embedding.onnx` | Extended segmentation model with speaker embeddings as an additional output (generated via `conversion/speech_embedding_export.py`)       |
| `models/ecapa_tdnn.onnx`           | ECAPA-TDNN speaker embedding model (192-dim), exported from [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| `models/ecapa_norm_mean.npy`       | Global embedding normalization mean for ECAPA-TDNN (192,)                                                                                |

## Usage

### Basic diarization

Outputs detected speakers with timestamps and confidence scores:

```bash
uv run python speech_diarizer.py
```

Automatically downloads the model and sample audio (`mlk.wav`), then prints
segments like:

```
SPEAKER_01      0.37s -    2.84s  (conf=0.951)
SPEAKER_02      2.84s -    5.21s  (conf=0.876)
```

### Diarization with embeddings

Extract per-segment speaker embeddings alongside timestamps:

```bash
uv run python speech_embedding.py
```

Output includes embedding dimensions for each segment, useful for downstream
clustering or verification.

### One-shot speaker diarization

Full pipeline: generate TTS voice references, extract ECAPA-TDNN embeddings,
segment audio with pyannote, and match segments to known speakers via
1-shot cosine similarity:

```bash
uv run python one_shot_diarization.py
```

This demo generates a multi-speaker conversation using KittenTTS (Bella + Bruno),
detects speech turns with the segmentation model, and assigns each turn to the
closest reference speaker using ECAPA-TDNN embeddings. A control voice (Luna)
is included to verify that unknown speakers are correctly rejected.

### Export segmentation model with embeddings

Re-exports the base ONNX model to include the LeakyRelu activation (speaker
embeddings) as a graph output:

```bash
uv run python conversion/speech_embedding_export.py
# Produces: models/model_with_embedding.onnx
```

### Export ECAPA-TDNN to ONNX

Export the SpeechBrain ECAPA-TDNN speaker embedding model to ONNX for
standalone inference:

```bash
uv run python conversion/ecapa_tdnn_export.py
# Produces: models/ecapa_tdnn.onnx, models/ecapa_norm_mean.npy
```

## Tests & Verification

### ECAPA-TDNN ONNX verification

Verifies that the exported ONNX ECAPA-TDNN model produces identical embeddings
to the original PyTorch model:

```bash
uv run python conversion/test_ecapa_onnx.py
```

Checks cosine similarity (>0.99) and max/mean absolute difference between
PyTorch and ONNX outputs on `mlk.wav`.

### Embedding segment quality analysis

Analyzes the quality of speaker embeddings extracted from the segmentation
model's embedding output:

```bash
uv run python conversion/test_embedding_segments.py
```

Computes:
- Segment-level and frame-level cosine similarity matrices
- Same-speaker vs. different-speaker similarity statistics
- Per-speaker within-speaker vs. between-speaker similarity

Verifies that same-speaker embeddings are more similar than different-speaker
embeddings at both segment and frame levels.

## Setup

Dependencies are managed via `uv`:

```bash
pip install uv          # if not already installed
uv sync                 # installs dependencies from pyproject.toml
```
