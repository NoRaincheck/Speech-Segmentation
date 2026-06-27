"""Speaker diarization pipeline combining segmentation and embedding."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from speech_segmentation.embedding import SpeakerEmbedder
from speech_segmentation.segmentation import SpeechSegmenter

FRAME_STEP = 270
TARGET_SR = 16000
MIN_SEGMENT_SAMPLES = 8000


@dataclass
class DiarizationMatch:
    """A segment matched to a known speaker."""

    speaker: str
    spk_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    similarity: float
    all_sims: dict[str, float] = field(default_factory=dict)


class Diarizer:
    """Full diarization pipeline: segment audio and match to known speakers.

    Args:
        segmenter: SpeechSegmenter instance.
        embedder: SpeakerEmbedder instance.
    """

    def __init__(self, segmenter: SpeechSegmenter, embedder: SpeakerEmbedder) -> None:
        self.segmenter = segmenter
        self.embedder = embedder

    def build_references(self, ref_embeddings: dict[str, np.ndarray]) -> None:
        """Set reference speaker embeddings for matching.

        Args:
            ref_embeddings: Mapping of speaker name to L2-normalized embedding.
        """
        self._ref_names = list(ref_embeddings.keys())
        self._ref_matrix = np.array([ref_embeddings[n] for n in self._ref_names])

    def diarize(self, audio_16k: np.ndarray) -> tuple[list[DiarizationMatch], list]:
        """Segment audio and match each segment to known speakers.

        Args:
            audio_16k: Audio samples at 16kHz, float32.

        Returns:
            Tuple of (list of DiarizationMatch, list of raw Segments).
        """
        segments = self.segmenter.segment(audio_16k)
        matches = self._match_segments(audio_16k, segments)
        return matches, segments

    def _match_segments(self, audio_16k: np.ndarray, segments: list) -> list[DiarizationMatch]:
        matches: list[DiarizationMatch] = []
        for seg in segments:
            seg_emb = self._compute_segment_embedding(audio_16k, seg.start_frame, seg.end_frame)
            if seg_emb is None:
                continue
            sims = self._cosine_similarity_matrix(seg_emb)
            best_idx = sims.argmax()
            matches.append(
                DiarizationMatch(
                    speaker=self._ref_names[best_idx],
                    spk_id=seg.speaker_id,
                    start_frame=seg.start_frame,
                    end_frame=seg.end_frame,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    confidence=seg.confidence,
                    similarity=float(sims[best_idx]),
                    all_sims={name: float(s) for name, s in zip(self._ref_names, sims)},
                )
            )
        return matches

    def _compute_segment_embedding(self, audio_16k: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray | None:
        start_sample = start_frame * FRAME_STEP
        end_sample = end_frame * FRAME_STEP
        segment_audio = audio_16k[start_sample:end_sample]
        if len(segment_audio) < MIN_SEGMENT_SAMPLES:
            return None
        return self.embedder.embed(segment_audio)

    def _cosine_similarity_matrix(self, emb: np.ndarray) -> np.ndarray:
        return self._ref_matrix @ emb
