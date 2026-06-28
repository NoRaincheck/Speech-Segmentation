# Speech Segmentation / Speaker Identification

Speaker identification using ONNX models (ECAPA-TDNN embeddings + optional VAE latent space).

## Models

| Model                              | Description                                                                                                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `models/model.onnx`                | Base segmentation model from [onnx-community/pyannote-segmentation-3.0](https://huggingface.co/onnx-community/pyannote-segmentation-3.0) |
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

## Setup

Dependencies are managed via `uv`:

```bash
pip install uv          # if not already installed
uv sync                 # installs dependencies from pyproject.toml
```
