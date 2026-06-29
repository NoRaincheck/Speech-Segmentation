"""Speaker diarization pipeline combining segmentation and embedding."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from speech_segmentation.embedding import SpeakerEmbedder
from speech_segmentation.segmentation import SpeechSegmenter

FRAME_STEP = 270
TARGET_SR = 16000
MIN_SEGMENT_SAMPLES = 8000
MIN_EMBED_SAMPLES = 12800


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
        vae: Optional SpeakerVAE instance. When provided, all embeddings are
            projected through the VAE encoder before matching.
    """

    def __init__(
        self,
        segmenter: SpeechSegmenter,
        embedder: SpeakerEmbedder,
        vae: object | None = None,
    ) -> None:
        self.segmenter = segmenter
        self.embedder = embedder
        self._vae = vae

    def build_references(self, ref_embeddings: dict[str, np.ndarray]) -> None:
        """Set reference speaker embeddings for matching.

        Args:
            ref_embeddings: Mapping of speaker name to L2-normalized embedding.
        """
        self._ref_names = list(ref_embeddings.keys())
        self._raw_ref_matrix = np.array([ref_embeddings[n] for n in self._ref_names])
        if self._vae is not None:
            self._ref_matrix = self._vae.encode_batch(self._raw_ref_matrix)
        else:
            self._ref_matrix = self._raw_ref_matrix

    def diarize(
        self,
        audio_16k: np.ndarray,
        stitch_threshold: float | None = None,
        stitch_raw: bool = False,
    ) -> tuple[list[DiarizationMatch], list]:
        """Segment audio and match each segment to known speakers.

        Args:
            audio_16k: Audio samples at 16kHz, float32.
            stitch_threshold: When set, consecutive segments with the same
                speaker label whose raw ECAPA-TDNN embeddings have cosine
                similarity above this threshold are merged.
            stitch_raw: When True, stitching uses raw ECAPA-TDNN embeddings
                (before VAE projection) for speaker label determination.
                When False, uses VAE-projected embeddings like the main matching.

        Returns:
            Tuple of (list of DiarizationMatch, list of raw Segments).
        """
        raw_segments = self.segmenter.segment(audio_16k)
        segments = self._filter_short_segments(raw_segments)
        matches = self._match_segments(audio_16k, segments)
        if stitch_threshold is not None:
            matches = self._stitch_segments(audio_16k, matches, stitch_threshold, stitch_raw)
        return matches, raw_segments

    def _filter_short_segments(self, segments: list) -> list:
        """Drop segments shorter than MIN_EMBED_SAMPLES.

        ECAPA-TDNN needs ~1s of speech for a reliable speaker embedding.
        Very short segments produce noisy embeddings that hurt matching.
        """
        return [s for s in segments if (s.end_frame - s.start_frame) * FRAME_STEP >= MIN_EMBED_SAMPLES]

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
            segment_audio = self._pad_and_repeat(segment_audio)
        emb = self.embedder.embed(segment_audio)
        if self._vae is not None:
            emb = self._vae.encode(emb)
        return emb

    def _stitch_segments(
        self,
        audio_16k: np.ndarray,
        matches: list[DiarizationMatch],
        threshold: float,
        use_raw: bool = False,
    ) -> list[DiarizationMatch]:
        """Merge consecutive same-speaker segments when embeddings are similar.

        Uses raw ECAPA-TDNN embeddings (before any VAE projection) to compute
        cosine similarity between adjacent segments. When similarity exceeds
        the threshold and the predicted speaker matches, segments are merged.

        Args:
            use_raw: When True, uses raw embeddings for speaker label
                determination. When False, uses the speaker labels from
                the original matching step.
        """
        if len(matches) <= 1:
            return matches

        merged: list[DiarizationMatch] = [matches[0]]
        for m in matches[1:]:
            prev = merged[-1]

            emb_prev = self._compute_raw_embedding(audio_16k, prev.start_frame, prev.end_frame)
            emb_curr = self._compute_raw_embedding(audio_16k, m.start_frame, m.end_frame)
            if emb_prev is None or emb_curr is None:
                merged.append(m)
                continue

            if use_raw:
                raw_sims_prev = self._raw_ref_matrix @ emb_prev
                prev_speaker = self._ref_names[raw_sims_prev.argmax()]
                raw_sims_curr = self._raw_ref_matrix @ emb_curr
                curr_speaker = self._ref_names[raw_sims_curr.argmax()]
            else:
                prev_speaker = prev.speaker
                curr_speaker = m.speaker

            if prev_speaker != curr_speaker:
                merged.append(m)
                continue

            sim = float(np.dot(emb_prev, emb_curr))
            if sim > threshold:
                merged[-1] = DiarizationMatch(
                    speaker=prev_speaker,
                    spk_id=prev.spk_id,
                    start_frame=prev.start_frame,
                    end_frame=m.end_frame,
                    start_time=prev.start_time,
                    end_time=m.end_time,
                    confidence=max(prev.confidence, m.confidence),
                    similarity=sim,
                    all_sims={**prev.all_sims, **m.all_sims},
                )
            else:
                merged.append(m)

        return merged

    def _compute_raw_embedding(self, audio_16k: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray | None:
        """Compute embedding without VAE projection for stitching."""
        start_sample = start_frame * FRAME_STEP
        end_sample = end_frame * FRAME_STEP
        segment_audio = audio_16k[start_sample:end_sample]
        if len(segment_audio) < MIN_SEGMENT_SAMPLES:
            segment_audio = self._pad_and_repeat(segment_audio)
        return self.embedder.embed(segment_audio)

    def _pad_and_repeat(self, audio: np.ndarray) -> np.ndarray:
        """Pad short audio by repeating the segment until MIN_SEGMENT_SAMPLES.

        Repeats the segment's own audio instead of padding with silence,
        which would corrupt the speaker embedding.
        """
        repeats = MIN_SEGMENT_SAMPLES // len(audio) + 1
        return np.tile(audio, repeats)[:MIN_SEGMENT_SAMPLES]

    def _cosine_similarity_matrix(self, emb: np.ndarray) -> np.ndarray:
        return self._ref_matrix @ emb
