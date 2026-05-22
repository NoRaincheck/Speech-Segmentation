import urllib.request
import wave
import numpy as np
import onnxruntime as ort

model_id = "onnx-community/pyannote-segmentation-3.0"

model_path = "model.onnx"
urllib.request.urlretrieve(
    f"https://huggingface.co/{model_id}/resolve/main/onnx/model.onnx",
    model_path,
)

url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/mlk.wav"
audio_path = "mlk.wav"
urllib.request.urlretrieve(url, audio_path)

with wave.open(audio_path, "rb") as wf:
    sr = wf.getframerate()
    frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

target_sr = 16000
step = 270

if sr != target_sr:
    old_len = len(audio)
    new_len = int(old_len * target_sr / sr)
    audio = np.interp(np.linspace(0, old_len - 1, new_len), np.arange(old_len), audio)

session = ort.InferenceSession(model_path)
logits = session.run(
    None, {"input_values": audio[np.newaxis, np.newaxis, :].astype(np.float32)}
)[0]

frame_logits = logits[0]
exps = np.exp(frame_logits - frame_logits.max(axis=1, keepdims=True))
probs = exps / exps.sum(axis=1, keepdims=True)

preds = probs.argmax(axis=1)
confidence = probs.max(axis=1)

segments = []
current_spk = None
current_start = None
max_conf = 0.0

for i, (cls, conf) in enumerate(zip(preds, confidence)):
    if cls in (1, 2, 3):
        if cls != current_spk:
            if current_spk is not None:
                segments.append(
                    (
                        current_spk,
                        current_start * step / target_sr,
                        i * step / target_sr,
                        float(max_conf),
                    )
                )
            current_spk = cls
            current_start = i
            max_conf = conf
        else:
            max_conf = max(max_conf, conf)
    else:
        if current_spk is not None:
            segments.append(
                (
                    current_spk,
                    current_start * step / target_sr,
                    i * step / target_sr,
                    float(max_conf),
                )
            )
            current_spk = None
            current_start = None
            max_conf = 0.0

if current_spk is not None:
    segments.append(
        (
            current_spk,
            current_start * step / target_sr,
            len(preds) * step / target_sr,
            float(max_conf),
        )
    )

for spk_id, start, end, conf in segments:
    print(f"  SPEAKER_{spk_id:02d}  {start:7.2f}s - {end:7.2f}s  (conf={conf:.3f})")
