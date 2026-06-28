# Speech Segmentation / Speaker Identification

Speaker identification using ONNX models (ECAPA-TDNN embeddings + optional VAE latent space).

## Models

| Model                              | Description                                                                                                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `models/model.onnx`                | Base segmentation model from [onnx-community/pyannote-segmentation-3.0](https://huggingface.co/onnx-community/pyannote-segmentation-3.0) |
| `models/model_with_embedding.onnx` | Extended segmentation model with speaker embeddings as an additional output (generated via `scripts/speech_embedding_export.py`)         |
| `models/ecapa_tdnn.onnx`           | ECAPA-TDNN speaker embedding model (192-dim), exported from [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| `models/ecapa_norm_mean.npy`       | Global embedding normalization mean for ECAPA-TDNN (192,)                                                                                |

## Usage

### Few-shot speaker identification

Generate TTS reference samples per speaker, build averaged speaker prototypes,
then classify individual dialogue turns via cosine similarity:

```bash
uv run python examples/speaker_identification.py
```

### Few-shot speaker identification with VAE

Same pipeline but trains a VAE on reference embeddings for improved speaker
separation in latent space:

```bash
uv run --group vae python examples/speaker_identification_vae.py
```

### Basic diarization

Segments audio into speaker turns using the pyannote ONNX model:

```bash
uv run python examples/basic_diarization.py
```

### Export segmentation model with embeddings

Re-exports the base ONNX model to include the LeakyRelu activation (speaker
embeddings) as a graph output:

```bash
uv run python scripts/speech_embedding_export.py
# Produces: models/model_with_embedding.onnx
```

### Export ECAPA-TDNN to ONNX

Export the SpeechBrain ECAPA-TDNN speaker embedding model to ONNX for
standalone inference:

```bash
uv run python scripts/ecapa_tdnn_export.py
# Produces: models/ecapa_tdnn.onnx, models/ecapa_norm_mean.npy
```

## Tests & Verification

### ECAPA-TDNN ONNX verification

Verifies that the exported ONNX ECAPA-TDNN model produces identical embeddings
to the original PyTorch model:

```bash
uv run python scripts/test_ecapa_onnx.py
```

Checks cosine similarity (>0.99) and max/mean absolute difference between
PyTorch and ONNX outputs on `mlk.wav`.

### Embedding segment quality analysis

Analyzes the quality of speaker embeddings extracted from the segmentation
model's embedding output:

```bash
uv run python scripts/test_embedding_segments.py
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
