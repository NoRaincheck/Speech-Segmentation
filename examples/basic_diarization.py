"""Basic speech diarization example.

Segments audio into speaker turns using the pyannote ONNX model.

Usage:
    uv run python examples/basic_diarization.py
"""

import os
import urllib.request

import numpy as np

from speech_segmentation import SpeechSegmenter

MODEL_ID = "onnx-community/pyannote-segmentation-3.0"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")

os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        f"https://huggingface.co/{MODEL_ID}/resolve/main/onnx/model.onnx",
        MODEL_PATH,
    )

url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/mlk.wav"
audio_path = "mlk.wav"
if not os.path.exists(audio_path):
    urllib.request.urlretrieve(url, audio_path)

import soundfile as sf

audio, sr = sf.read(audio_path, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
if sr != 16000:
    num_samples = int(len(audio) * 16000 / sr)
    audio = np.interp(np.linspace(0, len(audio) - 1, num_samples), np.arange(len(audio)), audio)

segmenter = SpeechSegmenter(MODEL_PATH)
segments = segmenter.segment(audio)

for seg in segments:
    print(f"  SPEAKER_{seg.speaker_id:02d}  {seg.start_time:7.2f}s - {seg.end_time:7.2f}s  (conf={seg.confidence:.3f})")
