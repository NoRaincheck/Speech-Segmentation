"""
One-Shot Speech Diarization Demo

Generates speech samples via KittenTTS, combines them into a conversation,
then uses 1-shot speaker embeddings to diarize and extract per-speaker audio.

Pipeline:
  1. Generate 3 reference utterances per voice with KittenTTS (Bella, Bruno, Luna)
  2. Extract ECAPA-TDNN speaker embeddings from each reference and average per voice
  3. Generate an 8-turn conversation using Bella and Bruno only
  4. Segment conversation with pyannote ONNX model (finds speech turns)
  5. Extract ECAPA-TDNN embedding for each segment
  6. Match each segment to the closest reference via cosine similarity
  7. Extract per-speaker audio into separate wav files

Models:
  - pyannote-segmentation-3.0 (ONNX) — speech segmentation (finds turns)
  - ECAPA-TDNN (ONNX) — speaker embedding extraction (192-dim)

Control: Luna's reference is extracted but never used in the conversation.
  This demonstrates that the system can reject unknown voices.

Usage:
    uv run python one_shot_diarization.py
"""

import os

import numpy as np
import onnxruntime as ort
import soundfile as sf
from kittentts import KittenTTS
from sklearn.metrics.pairwise import cosine_similarity

SEG_MODEL_PATH = "models/model.onnx"
EMB_MODEL_PATH = "models/ecapa_tdnn.onnx"
NORM_MEAN_PATH = "models/ecapa_norm_mean.npy"
TTS_SR = 24000
TARGET_SR = 16000
FBANK_SR = 16000
N_MELS = 80
FRAME_STEP = 270
SILENCE_DURATION = 0.5

REFERENCE_PHRASES = {
    "Bella": [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "The early bird catches the worm but the second mouse gets the cheese.",
    ],
    "Bruno": [
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump.",
        "The five boxing wizards jump quickly across the mat.",
    ],
    "Luna": [
        "She sells seashells by the seashore every Sunday morning.",
        "Peter Piper picked a peck of pickled peppers.",
        "Unique New York, unique New York, you know you need unique New York.",
    ],
}

DIALOGUE = [
    ("Bella", "Hey Bruno, have you tried the new coffee shop on Main Street?"),
    ("Bruno", "Yes, I went there yesterday. The espresso was really good."),
    ("Bella", "I heard they also have great pastries. I want to try the croissants."),
    ("Bruno", "They do. The almond croissant is my favorite. You should definitely go."),
    ("Bella", "Sounds perfect. Want to go together this weekend?"),
    ("Bruno", "Sure, I was thinking Saturday morning. Shall we meet at ten?"),
    ("Bella", "Ten works great for me. I'll see you there then."),
    ("Bruno", "Looking forward to it. Have a great day, Bella."),
]


def load_wav_16k(path):
    """Load audio and resample to 16kHz mono float32."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        num_samples = int(len(audio) * TARGET_SR / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, num_samples),
            np.arange(len(audio)),
            audio,
        )
    return audio


def extract_fbank(audio_16k):
    """Extract 80-dim Fbank features from 16kHz audio using numpy."""
    preemphasis = 0.97
    frame_length = 400
    frame_step = 160
    n_fft = 512

    audio = np.append(audio_16k[0], audio_16k[1:] - preemphasis * audio_16k[:-1])
    n_frames = 1 + (len(audio) - frame_length) // frame_step
    pad_length = n_frames * frame_step + frame_length - len(audio)
    audio = np.append(audio, np.zeros(pad_length))

    indices = np.arange(frame_length).reshape(1, -1) + np.arange(n_frames).reshape(-1, 1) * frame_step
    frames = audio[indices]
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1))
    frames *= hamming

    mag_frames = np.abs(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames**2)

    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (FBANK_SR / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, N_MELS + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / FBANK_SR).astype(int)

    n_freqs = n_fft // 2 + 1
    mel_fb = np.zeros((N_MELS, n_freqs))
    for m in range(1, N_MELS + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                mel_fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                mel_fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    fbank = np.dot(pow_frames, mel_fb.T)
    fbank = np.where(fbank == 0, np.finfo(float).eps, fbank)
    fbank = 10 * np.log10(fbank)
    return fbank


def segment_speech(probs):
    """Segment consecutive same-speaker frames from pyannote predictions."""
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    segments = []
    current_spk = None
    current_start = None
    max_conf = 0.0

    for i, (cls, c) in enumerate(zip(preds, conf)):
        if cls in (1, 2, 3):
            if cls != current_spk:
                if current_spk is not None:
                    segments.append((current_spk, current_start, i, float(max_conf)))
                current_spk = cls
                current_start = i
                max_conf = c
            else:
                max_conf = max(max_conf, c)
        else:
            if current_spk is not None:
                segments.append((current_spk, current_start, i, float(max_conf)))
                current_spk = None
                current_start = None
                max_conf = 0.0

    if current_spk is not None:
        segments.append((current_spk, current_start, len(preds), float(max_conf)))

    return segments


def run_segmentation(audio_16k, seg_session):
    """Run pyannote segmentation model to find speech segments."""
    logits = seg_session.run(None, {"input_values": audio_16k[np.newaxis, np.newaxis, :].astype(np.float32)})[0]
    frame_logits = logits[0]
    exps = np.exp(frame_logits - frame_logits.max(axis=1, keepdims=True))
    probs = exps / exps.sum(axis=1, keepdims=True)
    return segment_speech(probs)


def extract_embedding(audio_16k, emb_session, norm_mean):
    """Extract ECAPA-TDNN embedding from 16kHz audio.

    Returns a 192-dim L2-normalized embedding vector.
    """
    fbank = extract_fbank(audio_16k)
    raw_emb = emb_session.run(None, {"input_values": fbank[np.newaxis].astype(np.float32)})[0]
    emb = raw_emb.flatten() - norm_mean
    emb = emb / np.linalg.norm(emb)
    return emb


def compute_segment_embedding(audio_16k, start_frame, end_frame, emb_session, norm_mean):
    """Extract ECAPA-TDNN embedding for a specific segment of audio."""
    start_sample = start_frame * FRAME_STEP
    end_sample = end_frame * FRAME_STEP
    segment_audio = audio_16k[start_sample:end_sample]
    if len(segment_audio) < FBANK_SR * 0.5:
        return None
    return extract_embedding(segment_audio, emb_session, norm_mean)


def generate_reference_samples(tts, ref_dir):
    os.makedirs(ref_dir, exist_ok=True)
    ref_paths = {}
    for voice, phrases in REFERENCE_PHRASES.items():
        for j, phrase in enumerate(phrases):
            audio = tts.generate(phrase, voice=voice)
            path = os.path.join(ref_dir, f"{voice.lower()}_ref_{j}.wav")
            sf.write(path, audio, TTS_SR)
        ref_paths[voice] = [os.path.join(ref_dir, f"{voice.lower()}_ref_{j}.wav") for j in range(len(phrases))]
        print(f"  Generated {len(phrases)} reference files for {voice}")
    return ref_paths


def build_reference_embeddings(ref_paths, emb_session, norm_mean):
    """Build averaged ECAPA-TDNN embeddings per voice from reference files."""
    ref_embeddings = {}
    for voice, paths in ref_paths.items():
        voice_embs = []
        for path in paths:
            audio = load_wav_16k(path)
            emb = extract_embedding(audio, emb_session, norm_mean)
            voice_embs.append(emb)
            print(f"    {voice} {os.path.basename(path)}: emb norm={np.linalg.norm(emb):.4f}")
        avg_emb = np.mean(voice_embs, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        ref_embeddings[voice] = avg_emb
        print(f"  {voice}: averaged {len(voice_embs)} embeddings -> prototype norm={np.linalg.norm(avg_emb):.4f}")
    return ref_embeddings


def generate_conversation(tts, output_path):
    silence = np.zeros(int(TTS_SR * SILENCE_DURATION), dtype=np.float32)
    parts = []
    for i, (voice, text) in enumerate(DIALOGUE):
        audio = tts.generate(text, voice=voice)
        parts.append(audio)
        if i < len(DIALOGUE) - 1:
            parts.append(silence)
    conversation = np.concatenate(parts)
    sf.write(output_path, conversation, TTS_SR)
    print(f"  Generated {output_path} ({len(conversation) / TTS_SR:.2f}s, {len(DIALOGUE)} turns)")
    return conversation


def match_segments_to_references(conv_audio_16k, segments, ref_embeddings, emb_session, norm_mean):
    """Match each pyannote segment to the closest reference via ECAPA-TDNN embeddings."""
    ref_names = list(ref_embeddings.keys())
    ref_matrix = np.array([ref_embeddings[n] for n in ref_names])
    matches = []
    for spk_id, start, end, conf in segments:
        seg_emb = compute_segment_embedding(conv_audio_16k, start, end, emb_session, norm_mean)
        if seg_emb is None:
            continue
        sims = cosine_similarity(seg_emb.reshape(1, -1), ref_matrix)[0]
        best_idx = sims.argmax()
        start_t = start * FRAME_STEP / TARGET_SR
        end_t = end * FRAME_STEP / TARGET_SR
        matches.append(
            {
                "speaker": ref_names[best_idx],
                "spk_id": spk_id,
                "start_frame": start,
                "end_frame": end,
                "start_time": start_t,
                "end_time": end_t,
                "confidence": conf,
                "similarity": float(sims[best_idx]),
                "all_sims": {name: float(s) for name, s in zip(ref_names, sims)},
            }
        )
    return matches


def extract_per_speaker_audio(matches, conv_audio_24k):
    speaker_segments = {}
    for m in matches:
        name = m["speaker"]
        speaker_segments.setdefault(name, []).append(m)

    results = {}
    for name, segs in speaker_segments.items():
        parts = []
        for m in segs:
            start_16k = m["start_frame"] * FRAME_STEP
            end_16k = m["end_frame"] * FRAME_STEP
            start_24k = int(start_16k * TTS_SR / TARGET_SR)
            end_24k = int(end_16k * TTS_SR / TARGET_SR)
            start_24k = min(start_24k, len(conv_audio_24k))
            end_24k = min(end_24k, len(conv_audio_24k))
            if start_24k < end_24k:
                parts.append(conv_audio_24k[start_24k:end_24k])
        if parts:
            audio = np.concatenate(parts)
            path = f"{name.lower()}.wav"
            sf.write(path, audio, TTS_SR)
            results[name] = {"path": path, "duration": len(audio) / TTS_SR, "segments": len(segs)}
    return results


def main():
    tts = KittenTTS("KittenML/kitten-tts-nano-0.8")
    print("Loading ONNX models...")
    seg_session = ort.InferenceSession(SEG_MODEL_PATH)
    emb_session = ort.InferenceSession(EMB_MODEL_PATH)
    norm_mean = np.load(NORM_MEAN_PATH)
    print(f"  Segmentation model: {SEG_MODEL_PATH}")
    print(f"  Embedding model: {EMB_MODEL_PATH} (ECAPA-TDNN, 192-dim)")
    print(f"  Norm mean shape: {norm_mean.shape}")

    print("\n=== Step 1: Generating 1-shot reference samples ===")
    ref_paths = generate_reference_samples(tts, "refs")

    print("\n=== Step 2: Building 1-shot reference embeddings (ECAPA-TDNN) ===")
    ref_embeddings = build_reference_embeddings(ref_paths, emb_session, norm_mean)

    print("\n=== Step 3: Generating conversation ===")
    conv_audio = generate_conversation(tts, "conversation.wav")

    print("\n=== Step 4: Segmenting conversation (pyannote) ===")
    conv_audio_16k = load_wav_16k("conversation.wav")
    conv_segments = run_segmentation(conv_audio_16k, seg_session)
    print(f"  Detected {len(conv_segments)} speech segments")

    print("\n=== Step 5: 1-shot speaker matching (ECAPA-TDNN embeddings) ===")
    matches = match_segments_to_references(conv_audio_16k, conv_segments, ref_embeddings, emb_session, norm_mean)
    for i, m in enumerate(matches):
        sim_str = " ".join(f"{k}={v:.3f}" for k, v in m["all_sims"].items())
        print(
            f"  Seg {i:2d} ({m['start_time']:5.2f}s-{m['end_time']:5.2f}s): "
            f"{m['speaker']:5s} (sim={m['similarity']:.3f}) [{sim_str}]"
        )

    print(f"\n  --- Reference Similarity Summary ---")
    for name in REFERENCE_PHRASES:
        sims = [m["all_sims"][name] for m in matches]
        avg = np.mean(sims)
        is_control = name == "Luna"
        marker = " (control - should be low)" if is_control else ""
        print(f"  {name:6s} avg sim: {avg:.3f}{marker}")

    print("\n=== Step 6: Extracting per-speaker audio ===")
    conv_audio_24k, _ = sf.read("conversation.wav", dtype="float32")
    results = extract_per_speaker_audio(matches, conv_audio_24k)
    for name, info in sorted(results.items()):
        print(f"  {name}: {info['segments']} segments, {info['duration']:.1f}s total -> {info['path']}")

    print("\n=== Summary ===")
    for name in REFERENCE_PHRASES:
        segs_for_name = [m for m in matches if m["speaker"] == name]
        if segs_for_name:
            avg_sim = np.mean([m["similarity"] for m in segs_for_name])
            label = "CONTROL (should be absent)" if name == "Luna" else "SPEAKER"
            print(f"  {name:6s}: {len(segs_for_name):2d} segments, avg sim={avg_sim:.3f} [{label}]")
        else:
            label = "correctly absent" if name == "Luna" else "not detected"
            print(f"  {name:6s}: 0 segments [{label}]")

    expected_bella = sum(1 for v, _ in DIALOGUE if v == "Bella")
    expected_bruno = sum(1 for v, _ in DIALOGUE if v == "Bruno")
    print(f"\n  Expected dialogue turns: Bella={expected_bella}, Bruno={expected_bruno}")
    print(f"  pyannote segments may split/merge dialogue turns.")
    print(f"  ECAPA-TDNN provides 192-dim speaker embeddings for matching.")


if __name__ == "__main__":
    main()
