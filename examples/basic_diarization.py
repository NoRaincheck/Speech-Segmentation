"""Basic diarization: segment audio into speaker turns.

This example takes a multi-speaker conversation and answers the question
"who spoke when?" — without needing any reference audio or speaker labels.

It uses the pyannote-segmentation-3.0 ONNX model, which detects speech
regions and assigns each region to one of several internal speaker classes.
The output is a list of segments with timestamps and confidence scores.

Pipeline:
    audio → SpeechSegmenter → list of Segments (speaker_id, start, end)

Usage:
    uv run python examples/basic_diarization.py
    uv run python examples/basic_diarization.py --audio path/to/audio.wav
"""

import argparse
import time

from speech_segmentation import SpeechSegmenter
from speech_segmentation.audio import load_audio

# Path to the pyannote segmentation ONNX model.
# This model was exported from pyannote/segmentation-3.0 and runs inference
# in pure ONNX runtime (no PyTorch dependency at runtime).
MODEL_PATH = "models/model.onnx"


def main():
    parser = argparse.ArgumentParser(
        description="Diarize a conversation: detect who spoke when."
    )
    parser.add_argument(
        "--audio",
        default="combined_dialogue.wav",
        help="Path to a multi-speaker audio file (default: combined_dialogue.wav)",
    )
    args = parser.parse_args()

    # Step 1: Load and resample audio to 16kHz mono float32.
    # The segmentation model expects 16kHz input. If your audio is at a
    # different sample rate, load_audio handles resampling automatically.
    print(f"Loading {args.audio}...")
    audio, sr = load_audio(args.audio)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.1f}s, Sample rate: {sr}Hz")

    # Step 2: Load the segmentation model.
    # The ONNX model is ~6MB and loads in milliseconds.
    # It takes raw audio samples and outputs frame-level probabilities
    # for 4 classes: silence/overlap (0), speaker 1 (1), speaker 2 (2),
    # speaker 3 (3).
    print("Loading segmentation model...")
    t0 = time.perf_counter()
    segmenter = SpeechSegmenter(MODEL_PATH)
    print(f"  Loaded in {time.perf_counter() - t0:.2f}s")

    # Step 3: Segment the audio.
    # The model processes the audio in overlapping frames (270 samples ≈ 17ms
    # each), applies softmax to get per-frame speaker probabilities, then
    # groups consecutive same-speaker frames into segments.
    #
    # Returns a list of Segment objects, each with:
    #   - speaker_id: which speaker class (1, 2, or 3)
    #   - start_frame / end_frame: frame boundaries
    #   - start_time / end_time: computed timestamps in seconds
    #   - confidence: max probability across the segment
    print("Segmenting...")
    t0 = time.perf_counter()
    segments = segmenter.segment(audio)
    elapsed = time.perf_counter() - t0
    print(f"  Found {len(segments)} segments in {elapsed:.2f}s\n")

    # Step 4: Print results.
    # Each segment is a speaker turn. The speaker_id values (2, 3) are
    # internal class labels from the pyannote model — they don't correspond
    # to specific people. To know WHO is speaking, you'd need to match
    # segments against reference embeddings (see few_shot_classification.py).
    print(f"{'#':<4} {'Speaker':<10} {'Start':>8} {'End':>8} {'Duration':>10} {'Confidence':>10}")
    print("-" * 55)
    for i, seg in enumerate(segments):
        duration = seg.end_time - seg.start_time
        print(
            f"{i + 1:<4} {seg.speaker_id:<10} "
            f"{seg.start_time:>7.2f}s {seg.end_time:>7.2f}s "
            f"{duration:>9.2f}s {seg.confidence:>10.4f}"
        )

    # Summary: how much of the audio is speech vs silence?
    total_speech = sum(s.end_time - s.start_time for s in segments)
    print(f"\nSpeech: {total_speech:.2f}s / {duration:.2f}s ({total_speech / duration:.0%})")


if __name__ == "__main__":
    main()
