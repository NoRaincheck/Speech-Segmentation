"""Few-shot speaker identification example.

Generates reference utterances for each speaker, builds averaged speaker
prototypes, then generates individual dialogue turns and classifies each
one against the reference speakers via cosine matching.

Usage:
    uv run python examples/speaker_identification.py

Example output::

    Loading embedding model...

    === Step 1: Generating few-shot reference samples ===
      Reusing 3 existing reference files for Bella
      Reusing 3 existing reference files for Bruno
      Reusing 3 existing reference files for Luna

    === Step 2: Building few-shot reference embeddings (ECAPA-TDNN) ===
        Bella bella_ref_0.wav: emb norm=1.0000
        Bella bella_ref_1.wav: emb norm=1.0000
        Bella bella_ref_2.wav: emb norm=1.0000
      Bella: averaged 3 embeddings -> prototype norm=1.0000
        Bruno bruno_ref_0.wav: emb norm=1.0000
        Bruno bruno_ref_1.wav: emb norm=1.0000
        Bruno bruno_ref_2.wav: emb norm=1.0000
      Bruno: averaged 3 embeddings -> prototype norm=1.0000
        Luna luna_ref_0.wav: emb norm=1.0000
        Luna luna_ref_1.wav: emb norm=1.0000
        Luna luna_ref_2.wav: emb norm=1.0000
      Luna: averaged 3 embeddings -> prototype norm=1.0000

    === Step 3: Generating dialogue turns ===
      Reusing turns/turn_00_bella.wav
      Reusing turns/turn_01_bruno.wav
      Reusing turns/turn_02_bella.wav
      Reusing turns/turn_03_bruno.wav
      Reusing turns/turn_04_bella.wav
      Reusing turns/turn_05_bruno.wav
      Reusing turns/turn_06_bella.wav
      Reusing turns/turn_07_bruno.wav

    === Step 4: Classifying each turn against reference speakers ===

      Ground Truth   Predicted    Sim  Similarities
      ------------  ----------  -----  ------------------------------
             Bella       Bella  0.993  [Bella=0.993 Bruno=0.952 Luna=0.984]
             Bruno       Bruno  0.988  [Bella=0.939 Bruno=0.988 Luna=0.941]
             Bella       Bella  0.995  [Bella=0.995 Bruno=0.961 Luna=0.982]
             Bruno       Bruno  0.985  [Bella=0.938 Bruno=0.985 Luna=0.940]
             Bella       Bella  0.990  [Bella=0.990 Bruno=0.959 Luna=0.978]
             Bruno       Bruno  0.988  [Bella=0.954 Bruno=0.988 Luna=0.954]
             Bella       Bella  0.983  [Bella=0.983 Bruno=0.959 Luna=0.977]
             Bruno       Bruno  0.984  [Bella=0.953 Bruno=0.984 Luna=0.953]

      Accuracy: 8/8 turns correct (100.0%)
"""

import os

import numpy as np
import soundfile as sf
from kittentts import KittenTTS

from speech_segmentation import SpeakerEmbedder

EMB_MODEL_PATH = "models/ecapa_tdnn.onnx"
NORM_MEAN_PATH = "models/ecapa_norm_mean.npy"
TTS_SR = 24000

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
        paths = [os.path.join(ref_dir, f"{voice.lower()}_ref_{j}.wav") for j in range(len(phrases))]
        ref_paths[voice] = paths
        if all(os.path.exists(p) for p in paths):
            print(f"  Reusing {len(phrases)} existing reference files for {voice}")
            continue
        for j, phrase in enumerate(phrases):
            audio = tts.generate(phrase, voice=voice)
            sf.write(paths[j], audio, TTS_SR)
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
            emb = embedder.embed(audio)
            voice_embs.append(emb)
            print(f"    {voice} {os.path.basename(path)}: emb norm={np.linalg.norm(emb):.4f}")
        avg_emb = np.mean(voice_embs, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        ref_embeddings[voice] = avg_emb
        print(f"  {voice}: averaged {len(voice_embs)} embeddings -> prototype norm={np.linalg.norm(avg_emb):.4f}")
    return ref_embeddings


def generate_dialogue_turns(tts, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    turn_paths = []
    for i, (voice, text) in enumerate(DIALOGUE):
        path = os.path.join(output_dir, f"turn_{i:02d}_{voice.lower()}.wav")
        if os.path.exists(path):
            print(f"  Reusing {path}")
            turn_paths.append(path)
            continue
        audio = tts.generate(text, voice=voice)
        sf.write(path, audio, TTS_SR)
        turn_paths.append(path)
        print(f"  Generated {path} ({len(audio) / TTS_SR:.2f}s)")
    return turn_paths


def classify_turns(turn_paths, embedder, ref_embeddings):
    ref_names = list(ref_embeddings.keys())
    ref_matrix = np.array([ref_embeddings[n] for n in ref_names])

    print(f"\n  {'Ground Truth':>12s}  {'Predicted':>10s}  {'Sim':>5s}  Similarities")
    print(f"  {'-' * 12}  {'-' * 10}  {'-' * 5}  {'-' * 30}")

    correct = 0
    for i, path in enumerate(turn_paths):
        gt_voice = DIALOGUE[i][0]
        audio, _ = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        emb = embedder.embed(audio)
        emb = emb / np.linalg.norm(emb)
        sims = ref_matrix @ emb
        best_idx = int(sims.argmax())
        pred = ref_names[best_idx]
        is_correct = pred == gt_voice
        if is_correct:
            correct += 1
        mark = "" if is_correct else " [WRONG]"
        sim_str = " ".join(f"{n}={s:.3f}" for n, s in zip(ref_names, sims))
        print(f"  {gt_voice:>12s}  {pred:>10s}  {sims[best_idx]:5.3f}  [{sim_str}]{mark}")

    print(f"\n  Accuracy: {correct}/{len(turn_paths)} turns correct ({correct / len(turn_paths) * 100:.1f}%)")
    return correct, len(turn_paths)


def main():
    tts = KittenTTS("KittenML/kitten-tts-nano-0.8")
    print("Loading embedding model...")
    embedder = SpeakerEmbedder(EMB_MODEL_PATH, NORM_MEAN_PATH)

    print("\n=== Step 1: Generating few-shot reference samples ===")
    ref_paths = generate_reference_samples(tts, "refs")

    print("\n=== Step 2: Building few-shot reference embeddings (ECAPA-TDNN) ===")
    ref_embeddings = build_reference_embeddings(ref_paths, embedder)

    print("\n=== Step 3: Generating dialogue turns ===")
    turn_paths = generate_dialogue_turns(tts, "turns")

    print("\n=== Step 4: Classifying each turn against reference speakers ===")
    classify_turns(turn_paths, embedder, ref_embeddings)


if __name__ == "__main__":
    main()
