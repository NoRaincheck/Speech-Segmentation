import wave
import numpy as np
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity

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

# Collect segment-level embeddings and frame-level embeddings with speaker labels
segment_embeddings = []
speaker_labels = []
frame_embeddings_with_speakers = []  # (embedding, speaker_id)

print("=" * 60)
print("SEGMENT EMBEDDINGS")
print("=" * 60)

for spk_id, start_frame, end_frame, conf in segments:
    seg_embeddings = frame_embs[start_frame:end_frame]
    mean_emb = seg_embeddings.mean(axis=0)
    start_time = start_frame * step / target_sr
    end_time = end_frame * step / target_sr
    print(
        f"  SPEAKER_{spk_id:02d}  {start_time:7.2f}s - {end_time:7.2f}s  (conf={conf:.3f})  emb_dim={mean_emb.shape}"
    )
    segment_embeddings.append(mean_emb)
    speaker_labels.append(spk_id)
    
    # Also collect frame-level embeddings
    for i in range(start_frame, end_frame):
        frame_embeddings_with_speakers.append((frame_embs[i], spk_id))

print()

# Compute segment-level similarity matrix
if len(segment_embeddings) > 0:
    seg_emb_array = np.array(segment_embeddings)
    seg_similarity = cosine_similarity(seg_emb_array)
    
    print("=" * 60)
    print("SEGMENT-LEVEL SIMILARITY MATRIX (Cosine Similarity)")
    print("=" * 60)
    print("\nColumns/Rows correspond to segments in order:")
    for i, (spk_id, _, _, _) in enumerate(segments):
        print(f"  Segment {i}: SPEAKER_{spk_id:02d}")
    
    print("\nSimilarity Matrix:")
    print(seg_similarity)
    
    # Analyze segment-level similarity
    print("\n" + "=" * 60)
    print("SEGMENT-LEVEL ANALYSIS")
    print("=" * 60)
    
    n_segments = len(segments)
    same_speaker_sim = []
    different_speaker_sim = []
    
    for i in range(n_segments):
        for j in range(n_segments):
            if i == j:
                # Diagonal (self-similarity)
                same_speaker_sim.append(seg_similarity[i, j])
            elif speaker_labels[i] == speaker_labels[j]:
                # Same speaker, different segment
                same_speaker_sim.append(seg_similarity[i, j])
            else:
                # Different speaker
                different_speaker_sim.append(seg_similarity[i, j])
    
    print(f"Average same-speaker similarity: {np.mean(same_speaker_sim):.4f}")
    print(f"Average different-speaker similarity: {np.mean(different_speaker_sim):.4f}")
    print(f"Mean difference (same - different): {np.mean(same_speaker_sim) - np.mean(different_speaker_sim):.4f}")
    
    if np.mean(same_speaker_sim) > np.mean(different_speaker_sim):
        print("\n✓ VERIFIED: Same-speaker embeddings are more similar than different-speaker embeddings")
    else:
        print("\n✗ WARNING: Same-speaker embeddings are NOT more similar (segment-level)")

print()
print("=" * 60)
print("FRAME-LEVEL ANALYSIS")
print("=" * 60)

# Collect frame embeddings and their speaker labels
frame_embs_array = np.array([emb for emb, _ in frame_embeddings_with_speakers])
frame_labels = np.array([label for _, label in frame_embeddings_with_speakers])

# Compute frame-level similarity matrix
frame_similarity = cosine_similarity(frame_embs_array)

print(f"Total frames: {len(frame_labels)}")
print(f"Unique speakers: {np.unique(frame_labels)}")

# Analyze frame-level similarity
same_speaker_frame_sim = []
different_speaker_frame_sim = []

for i in range(len(frame_labels)):
    for j in range(len(frame_labels)):
        if i == j:
            continue  # Skip self-similarity
        elif frame_labels[i] == frame_labels[j]:
            same_speaker_frame_sim.append(frame_similarity[i, j])
        else:
            different_speaker_frame_sim.append(frame_similarity[i, j])

print(f"Average same-speaker frame similarity: {np.mean(same_speaker_frame_sim):.4f}")
print(f"Average different-speaker frame similarity: {np.mean(different_speaker_frame_sim):.4f}")
print(f"Mean difference (same - different): {np.mean(same_speaker_frame_sim) - np.mean(different_speaker_frame_sim):.4f}")

if np.mean(same_speaker_frame_sim) > np.mean(different_speaker_frame_sim):
    print("\n✓ VERIFIED: Same-speaker frame embeddings are more similar than different-speaker embeddings")
else:
    print("\n✗ WARNING: Same-speaker frame embeddings are NOT more similar (frame-level)")

# Print a sample of the frame-level similarity matrix (first 50 frames for readability)
print("\n" + "=" * 60)
print("SAMPLE FRAME-LEVEL SIMILARITY MATRIX (First 50 frames)")
print("=" * 60)
sample_size = min(50, len(frame_labels))
sample_similarity = frame_similarity[:sample_size, :sample_size]
sample_labels = frame_labels[:sample_size]

print("\nSpeaker labels for each frame:")
for i in range(sample_size):
    print(f"Frame {i:2d}: SPEAKER_{sample_labels[i]:02d}")

print("\nSimilarity Matrix (first 50 frames):")
print(sample_similarity)

# Additional verification: compute within-speaker vs between-speaker averages per speaker
print("\n" + "=" * 60)
print("PER-SPEAKER STATISTICS")
print("=" * 60)

unique_speakers = np.unique(frame_labels)
for spk in unique_speakers:
    indices = np.where(frame_labels == spk)[0]
    if len(indices) < 2:
        continue
    
    # Within-speaker similarity (excluding diagonal)
    within_sim = []
    for i in range(len(indices)):
        for j in range(len(indices)):
            if i != j:
                within_sim.append(frame_similarity[indices[i], indices[j]])
    
    # Between-speaker similarity
    between_sim = []
    for i in indices:
        for j in range(len(frame_labels)):
            if frame_labels[j] != spk:
                between_sim.append(frame_similarity[i, j])
    
    print(f"\nSPEAKER_{spk:02d}:")
    print(f"  Frames: {len(indices)}")
    print(f"  Within-speaker similarity (avg): {np.mean(within_sim):.4f}")
    print(f"  Between-speaker similarity (avg): {np.mean(between_sim):.4f}")
