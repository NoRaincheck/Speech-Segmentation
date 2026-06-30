"""Diarization demo: segment a conversation into speaker turns.

Loads a multi-speaker conversation, runs pyannote-based speech segmentation
to detect who speaks when, then prints each detected turn with timestamps.

Usage:
    uv run python demos/diarization.py
    uv run python demos/diarization.py --audio path/to/audio.wav
"""

import argparse
import time

import numpy as np

from speech_segmentation import SpeechSegmenter
from speech_segmentation.audio import load_audio

MODELS_DIR = "models"


def main():
    parser = argparse.ArgumentParser(description="Speech diarization demo")
    parser.add_argument("--audio", default="combined_dialogue.wav", help="Path to audio file")
    args = parser.parse_args()

    print("Loading audio...")
    audio, sr = load_audio(args.audio)
    print(f"  Duration: {len(audio) / sr:.1f}s, Sample rate: {sr}Hz")

    print("Loading segmentation model...")
    t0 = time.perf_counter()
    segmenter = SpeechSegmenter(f"{MODELS_DIR}/model.onnx")
    print(f"  Loaded in {time.perf_counter() - t0:.2f}s")

    print("Segmenting audio...")
    t0 = time.perf_counter()
    segments = segmenter.segment(audio)
    elapsed = time.perf_counter() - t0
    print(f"  Found {len(segments)} segments in {elapsed:.2f}s\n")

    print(f"{'#':<4} {'Speaker':<10} {'Start':>8} {'End':>8} {'Duration':>10} {'Confidence':>10}")
    print("-" * 55)
    for i, seg in enumerate(segments):
        duration = seg.end_time - seg.start_time
        print(
            f"{i + 1:<4} {seg.speaker_id:<10} {seg.start_time:>7.2f}s {seg.end_time:>7.2f}s "
            f"{duration:>9.2f}s {seg.confidence:>10.4f}"
        )

    total_speech = sum(s.end_time - s.start_time for s in segments)
    print(f"\nTotal speech: {total_speech:.2f}s / {len(audio) / sr:.2f}s audio")


if __name__ == "__main__":
    main()
