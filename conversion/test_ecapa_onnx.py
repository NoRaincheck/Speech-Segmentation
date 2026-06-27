"""
Verify that the ONNX ECAPA-TDNN model produces identical output to PyTorch.

Usage (from project root):
    python conversion/test_ecapa_onnx.py
"""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

PROJ_DIR = Path(__file__).resolve().parent.parent
AUDIO_PATH = PROJ_DIR / "mlk.wav"
ONNX_PATH = PROJ_DIR / "models" / "ecapa_tdnn.onnx"
NORM_PATH = PROJ_DIR / "models" / "ecapa_norm_mean.npy"


def extract_fbank_torchaudio(wav_path, sample_rate=16000, n_mels=80):
    audio, sr = sf.read(wav_path, dtype="float32")
    if sr != sample_rate:
        num_samples = int(len(audio) * sample_rate / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, num_samples),
            np.arange(len(audio)),
            audio,
        )
    waveform = torch.tensor(audio).unsqueeze(0)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=n_mels,
        sample_frequency=sample_rate,
    )
    return fbank.unsqueeze(0)


def main():
    print("=" * 60)
    print("ECAPA-TDNN ONNX Verification")
    print("=" * 60)

    print(f"\n[1] Loading PyTorch model via SpeechBrain...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    ecapa_model = classifier.mods["embedding_model"]
    ecapa_model.eval()
    mean_var_norm_emb = classifier.mods["mean_var_norm_emb"]

    print(f"[2] Loading ONNX model from {ONNX_PATH}...")
    session = ort.InferenceSession(str(ONNX_PATH))
    norm_mean = np.load(NORM_PATH)

    print(f"[3] Extracting Fbank features from {AUDIO_PATH}...")
    fbank = extract_fbank_torchaudio(AUDIO_PATH)
    print(f"    Fbank shape: {fbank.shape}")

    print("[4] Running PyTorch inference...")
    start = time.perf_counter()
    with torch.no_grad():
        torch_emb = ecapa_model(fbank, lengths=None)
        torch_emb = mean_var_norm_emb(torch_emb, torch.ones(1))
    torch_time = time.perf_counter() - start
    torch_emb_np = torch_emb.numpy().flatten()
    print(f"    Embedding shape: {torch_emb.shape}, time: {torch_time * 1000:.1f}ms")

    print("[5] Running ONNX inference...")
    start = time.perf_counter()
    onnx_raw = session.run(None, {"input_values": fbank.numpy()})[0]
    onnx_emb = onnx_raw.flatten() - norm_mean
    onnx_emb = onnx_emb / np.linalg.norm(onnx_emb)
    onnx_time = time.perf_counter() - start
    print(f"    Embedding shape: {onnx_raw.shape}, time: {onnx_time * 1000:.1f}ms")

    torch_emb_norm = torch_emb_np / np.linalg.norm(torch_emb_np)

    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    cos_sim = np.dot(torch_emb_norm, onnx_emb)
    max_diff = np.max(np.abs(torch_emb_np - onnx_raw.flatten() + norm_mean))
    mean_diff = np.mean(np.abs(torch_emb_np - onnx_raw.flatten() + norm_mean))

    print(f"  Cosine similarity (after norm):  {cos_sim:.6f}")
    print(f"  Max absolute difference:         {max_diff:.8f}")
    print(f"  Mean absolute difference:        {mean_diff:.8f}")
    print(f"  PyTorch time:                    {torch_time * 1000:.1f}ms")
    print(f"  ONNX time:                       {onnx_time * 1000:.1f}ms")

    if cos_sim > 0.99:
        print("\n  PASS: ONNX output matches PyTorch (cosine sim > 0.99)")
    else:
        print(f"\n  FAIL: Cosine similarity {cos_sim:.4f} < 0.99")


if __name__ == "__main__":
    main()
