"""One-shot speech diarization example.

Generates reference utterances, builds speaker prototypes, then diarizes
a conversation and matches segments to known speakers.

Usage:
    uv run python examples/one_shot_diarization.py
"""

import os

import numpy as np
import soundfile as sf
from kittentts import KittenTTS

from speech_segmentation import Diarizer, SpeakerEmbedder, SpeechSegmenter

SEG_MODEL_PATH = "models/model.onnx"
EMB_MODEL_PATH = "models/ecapa_tdnn.onnx"
NORM_MEAN_PATH = "models/ecapa_norm_mean.npy"
TTS_SR = 24000
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


def build_reference_embeddings(ref_paths, embedder):
    ref_embeddings = {}
    for voice, paths in ref_paths.items():
        voice_embs = []
        for path in paths:
            audio, _ = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if len(audio) != 16000 * len(audio) // 16000:
                num_samples = int(len(audio) * 16000 / 24000)
                audio = np.interp(np.linspace(0, len(audio) - 1, num_samples), np.arange(len(audio)), audio)
            emb = embedder.embed(audio)
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


def extract_per_speaker_audio(matches, conv_audio_24k):
    speaker_segments = {}
    for m in matches:
        name = m.speaker
        speaker_segments.setdefault(name, []).append(m)

    results = {}
    for name, segs in speaker_segments.items():
        parts = []
        for m in segs:
            start_16k = m.start_frame * 270
            end_16k = m.end_frame * 270
            start_24k = int(start_16k * TTS_SR / 16000)
            end_24k = int(end_16k * TTS_SR / 16000)
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
    segmenter = SpeechSegmenter(SEG_MODEL_PATH)
    embedder = SpeakerEmbedder(EMB_MODEL_PATH, NORM_MEAN_PATH)
    diarizer = Diarizer(segmenter, embedder)

    print("\n=== Step 1: Generating 1-shot reference samples ===")
    ref_paths = generate_reference_samples(tts, "refs")

    print("\n=== Step 2: Building 1-shot reference embeddings (ECAPA-TDNN) ===")
    ref_embeddings = build_reference_embeddings(ref_paths, embedder)
    diarizer.build_references(ref_embeddings)

    print("\n=== Step 3: Generating conversation ===")
    conv_audio = generate_conversation(tts, "conversation.wav")

    print("\n=== Step 4: Segmenting conversation (pyannote) ===")
    conv_audio_16k, _ = sf.read("conversation.wav", dtype="float32")
    if conv_audio_16k.ndim > 1:
        conv_audio_16k = conv_audio_16k.mean(axis=1)

    print("\n=== Step 5: 1-shot speaker matching (ECAPA-TDNN embeddings) ===")
    matches, segments = diarizer.diarize(conv_audio_16k)
    print(f"  Detected {len(segments)} speech segments")
    for i, m in enumerate(matches):
        sim_str = " ".join(f"{k}={v:.3f}" for k, v in m.all_sims.items())
        print(
            f"  Seg {i:2d} ({m.start_time:5.2f}s-{m.end_time:5.2f}s): "
            f"{m.speaker:5s} (sim={m.similarity:.3f}) [{sim_str}]"
        )

    print("\n  --- Reference Similarity Summary ---")
    for name in REFERENCE_PHRASES:
        sims = [m.all_sims[name] for m in matches]
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
        segs_for_name = [m for m in matches if m.speaker == name]
        if segs_for_name:
            avg_sim = np.mean([m.similarity for m in segs_for_name])
            label = "CONTROL (should be absent)" if name == "Luna" else "SPEAKER"
            print(f"  {name:6s}: {len(segs_for_name):2d} segments, avg sim={avg_sim:.3f} [{label}]")
        else:
            label = "correctly absent" if name == "Luna" else "not detected"
            print(f"  {name:6s}: 0 segments [{label}]")


if __name__ == "__main__":
    main()
