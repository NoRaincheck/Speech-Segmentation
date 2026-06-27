"""Speech segmentation using pyannote-segmentation-3.0 ONNX model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

FRAME_STEP = 270
TARGET_SR = 16000


@dataclass
class Segment:
    """A detected speech segment."""

    speaker_id: int
    start_frame: int
    end_frame: int
    confidence: float

    @property
    def start_time(self) -> float:
        return self.start_frame * FRAME_STEP / TARGET_SR

    @property
    def end_time(self) -> float:
        return self.end_frame * FRAME_STEP / TARGET_SR


class SpeechSegmenter:
    """Segment audio into speaker turns using pyannote ONNX model.

    Args:
        model_path: Path to the pyannote ONNX model file.
    """

    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(model_path)

    def segment(self, audio_16k: np.ndarray) -> list[Segment]:
        """Segment 16kHz audio into speaker turns.

        Args:
            audio_16k: Audio samples at 16kHz, float32.

        Returns:
            List of detected speech segments.
        """
        logits = self.session.run(
            None, {"input_values": audio_16k[np.newaxis, np.newaxis, :].astype(np.float32)}
        )[0]
        frame_logits = logits[0]
        exps = np.exp(frame_logits - frame_logits.max(axis=1, keepdims=True))
        probs = exps / exps.sum(axis=1, keepdims=True)
        return _segment_speech(probs)


def _segment_speech(probs: np.ndarray) -> list[Segment]:
    """Segment consecutive same-speaker frames from pyannote predictions."""
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    segments: list[Segment] = []
    current_spk: int | None = None
    current_start: int | None = None
    max_conf = 0.0

    for i, (cls, c) in enumerate(zip(preds, conf)):
        if cls in (1, 2, 3):
            if cls != current_spk:
                if current_spk is not None:
                    segments.append(Segment(current_spk, current_start, i, float(max_conf)))
                current_spk = int(cls)
                current_start = i
                max_conf = c
            else:
                max_conf = max(max_conf, c)
        else:
            if current_spk is not None:
                segments.append(Segment(current_spk, current_start, i, float(max_conf)))
                current_spk = None
                current_start = None
                max_conf = 0.0

    if current_spk is not None:
        segments.append(Segment(current_spk, current_start, len(preds), float(max_conf)))

    return segments
