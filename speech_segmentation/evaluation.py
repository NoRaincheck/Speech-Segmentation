"""Evaluation metrics for speaker verification and diarization.

Provides EER (Equal Error Rate), minDCF (minimum Detection Cost Function),
DER (Diarization Error Rate), and cross-validation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VerificationMetrics:
    """Results from speaker verification evaluation."""

    eer: float
    min_dcf: float
    threshold_at_eer: float
    far_at_eer: float
    frr_at_eer: float


@dataclass
class DiarizationMetrics:
    """Results from diarization evaluation."""

    der: float
    miss_rate: float
    false_alarm_rate: float
    confusion_rate: float
    total_reference_speech: float
    total_hypothesis_speech: float


def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> VerificationMetrics:
    """Compute Equal Error Rate and related metrics.

    Args:
        genuine_scores: Cosine similarity scores for same-speaker pairs.
        impostor_scores: Cosine similarity scores for different-speaker pairs.

    Returns:
        VerificationMetrics with EER, minDCF, and operating point details.
    """
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    all_labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])

    thresholds = np.sort(np.unique(all_scores))
    if len(thresholds) == 0:
        return VerificationMetrics(1.0, 1.0, 0.5, 1.0, 1.0)

    far_list = []
    frr_list = []
    for t in thresholds:
        predicted_pos = all_scores >= t
        tp = np.sum((predicted_pos == 1) & (all_labels == 1))
        fp = np.sum((predicted_pos == 1) & (all_labels == 0))
        fn = np.sum((predicted_pos == 0) & (all_labels == 1))
        tn = np.sum((predicted_pos == 0) & (all_labels == 0))

        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer = (far_arr[eer_idx] + frr_arr[eer_idx]) / 2
    threshold_at_eer = thresholds[eer_idx]

    p_target = 0.5
    c_miss = 1.0
    c_fa = 1.0
    min_dcf = np.min(c_miss * frr_arr * p_target + c_fa * far_arr * (1 - p_target))

    return VerificationMetrics(
        eer=float(eer),
        min_dcf=float(min_dcf),
        threshold_at_eer=float(threshold_at_eer),
        far_at_eer=float(far_arr[eer_idx]),
        frr_at_eer=float(frr_arr[eer_idx]),
    )


def compute_diarization_error_rate(
    reference_segments: list[tuple[float, float, str]],
    hypothesis_segments: list[tuple[float, float, str]],
    collar: float = 0.25,
    overlap_threshold: float = 0.5,
) -> DiarizationMetrics:
    """Compute Diarization Error Rate (DER).

    DER = (miss + false_alarm + confusion) / total_reference_speech

    Args:
        reference_segments: List of (start_sec, end_sec, speaker_label).
        hypothesis_segments: List of (start_sec, end_sec, speaker_label).
        collar: Tolerance in seconds for boundary alignment.
        overlap_threshold: Minimum overlap ratio to count as a match.

    Returns:
        DiarizationMetrics with DER breakdown.
    """
    if not reference_segments:
        total_ref = sum(e - s for s, e, _ in hypothesis_segments)
        return DiarizationMetrics(1.0, 0.0, 1.0, 0.0, 0.0, total_ref)

    total_ref = sum(e - s for s, e, _ in reference_segments)
    total_hyp = sum(e - s for s, e, _ in hypothesis_segments)

    time_slots = set()
    for s, e, _ in reference_segments:
        time_slots.add(round(s, 3))
        time_slots.add(round(e, 3))
    for s, e, _ in hypothesis_segments:
        time_slots.add(round(s, 3))
        time_slots.add(round(e, 3))

    sorted_times = sorted(time_slots)
    miss_time = 0.0
    fa_time = 0.0
    conf_time = 0.0

    for i in range(len(sorted_times) - 1):
        t_start = sorted_times[i]
        t_end = sorted_times[i + 1]
        duration = t_end - t_start
        if duration <= 0:
            continue

        ref_speakers = set()
        for s, e, spk in reference_segments:
            if s <= t_start + collar and e >= t_end - collar:
                ref_speakers.add(spk)

        hyp_speakers = set()
        for s, e, spk in hypothesis_segments:
            if s <= t_start + collar and e >= t_end - collar:
                hyp_speakers.add(spk)

        if not ref_speakers and hyp_speakers:
            fa_time += duration
        elif ref_speakers and not hyp_speakers:
            miss_time += duration
        elif ref_speakers and hyp_speakers:
            if ref_speakers != hyp_speakers:
                conf_time += duration

    miss_rate = miss_time / total_ref if total_ref > 0 else 0.0
    fa_rate = fa_time / total_ref if total_ref > 0 else 0.0
    conf_rate = conf_time / total_ref if total_ref > 0 else 0.0
    der = miss_rate + fa_rate + conf_rate

    return DiarizationMetrics(
        der=float(der),
        miss_rate=float(miss_rate),
        false_alarm_rate=float(fa_rate),
        confusion_rate=float(conf_rate),
        total_reference_speech=float(total_ref),
        total_hypothesis_speech=float(total_hyp),
    )


def cross_validate(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 42,
) -> list[VerificationMetrics]:
    """Perform cross-validation for speaker verification.

    Args:
        embeddings: Array of shape (N, dim).
        labels: Speaker labels of shape (N,).
        n_folds: Number of folds.
        random_seed: Random seed for reproducibility.

    Returns:
        List of VerificationMetrics, one per fold.
    """
    rng = np.random.RandomState(random_seed)
    unique_labels = np.unique(labels)
    n_samples = len(embeddings)
    fold_indices = np.array_split(rng.permutation(n_samples), n_folds)

    metrics_list = []
    for fold_idx in range(n_folds):
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[fold_indices[fold_idx]] = True
        train_mask = ~test_mask

        train_embs = embeddings[train_mask]
        train_labels = labels[train_mask]
        test_embs = embeddings[test_mask]
        test_labels = labels[test_mask]

        if len(np.unique(test_labels)) < 2:
            continue

        genuine_scores = []
        impostor_scores = []
        for i in range(len(test_embs)):
            for j in range(len(train_embs)):
                sim = np.dot(test_embs[i], train_embs[j]) / (
                    np.linalg.norm(test_embs[i]) * np.linalg.norm(train_embs[j]) + 1e-8
                )
                if test_labels[i] == train_labels[j]:
                    genuine_scores.append(sim)
                else:
                    impostor_scores.append(sim)

        if not genuine_scores or not impostor_scores:
            continue

        metrics = compute_eer(np.array(genuine_scores), np.array(impostor_scores))
        metrics_list.append(metrics)

    return metrics_list


def print_verification_metrics(metrics: VerificationMetrics, prefix: str = "") -> None:
    """Pretty-print verification metrics."""
    print(f"{prefix}EER:           {metrics.eer:.4f} ({metrics.eer * 100:.2f}%)")
    print(f"{prefix}minDCF:        {metrics.min_dcf:.4f}")
    print(f"{prefix}Threshold@EER: {metrics.threshold_at_eer:.4f}")
    print(f"{prefix}FAR@EER:       {metrics.far_at_eer:.4f}")
    print(f"{prefix}FRR@EER:       {metrics.frr_at_eer:.4f}")


def print_diarization_metrics(metrics: DiarizationMetrics, prefix: str = "") -> None:
    """Pretty-print diarization metrics."""
    print(f"{prefix}DER:           {metrics.der:.4f} ({metrics.der * 100:.2f}%)")
    print(f"{prefix}Miss Rate:     {metrics.miss_rate:.4f}")
    print(f"{prefix}FA Rate:       {metrics.false_alarm_rate:.4f}")
    print(f"{prefix}Confusion:     {metrics.confusion_rate:.4f}")
    print(f"{prefix}Ref Speech:    {metrics.total_reference_speech:.2f}s")
    print(f"{prefix}Hyp Speech:    {metrics.total_hypothesis_speech:.2f}s")
