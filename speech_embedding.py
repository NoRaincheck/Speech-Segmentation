import wave
import numpy as np
import onnxruntime as ort

model_path = "model_with_embedding.onnx"
audio_path = "mlk.wav"

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
logits, embeddings = session.run(
    None, {"input_values": audio[np.newaxis, np.newaxis, :].astype(np.float32)}
)

frame_logits = logits[0]
frame_embs = embeddings[0]

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
                        current_start,
                        i,
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
                    current_start,
                    i,
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
            current_start,
            len(preds),
            float(max_conf),
        )
    )

for spk_id, start_frame, end_frame, conf in segments:
    seg_embeddings = frame_embs[start_frame:end_frame]
    mean_emb = seg_embeddings.mean(axis=0)
    start_time = start_frame * step / target_sr
    end_time = end_frame * step / target_sr
    print(
        f"  SPEAKER_{spk_id:02d}  {start_time:7.2f}s - {end_time:7.2f}s  (conf={conf:.3f})  emb_dim={mean_emb.shape}"
    )
