# Speech Segmentation / Speaker Identification

Speaker identification using ONNX models (ECAPA-TDNN embeddings + optional VAE latent space).

## Models

| Model                              | Description                                                                                                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `models/model.onnx`                | Base segmentation model from [onnx-community/pyannote-segmentation-3.0](https://huggingface.co/onnx-community/pyannote-segmentation-3.0) |
| `models/ecapa_tdnn.onnx`           | ECAPA-TDNN speaker embedding model (192-dim), exported from [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |

## Examples

Three well-commented examples demonstrating the core workflows:

```bash
# 1. Basic diarization — segment audio into speaker turns (no reference needed)
uv run python examples/basic_diarization.py

# 2. Few-shot classification — identify speakers from reference clips
uv run python examples/few_shot_classification.py

# 3. VAE cross-engine — when VAE improves matching across TTS engines
uv run --group vae python examples/vae_cross_engine.py
```

## Demos

Additional demos exploring ablations and edge cases:

```bash
uv run python demos/diarization.py
uv run python demos/few_shot_classification.py
uv run --group vae python demos/vae_ablation.py
uv run --group vae python demos/example_when_vae_helps.py
uv run python demos/example_multi_tts.py
```

See `demos/REPORT.md` for full results.

## Setup

```bash
uv sync
```
