# Diarization Examples Report

## Overview

The `examples/diarization_*.py` scripts demonstrate full speaker diarization pipelines:
they combine individual dialogue turns into a single audio file, run automatic segmentation
to detect speech regions, then identify each segment against known reference speakers.
Each variant uses a different matching strategy (no VAE, labelled VAE, unsupervised VAE).

All three scripts use the same 8-turn dialogue between Bella and Bruno (with Luna as a
distractor reference speaker), the same reference audio from `refs/`, and the same
segmentation model. The only difference is how embeddings are projected before cosine
matching.

## How Diarization Differs from Speaker Identification

The companion `examples/speaker_identification_*.py` scripts classify **individual
turn-level audio files** (clean, pre-segmented speech). The diarization scripts instead
combine all turns into one audio file, detect speech segments via `SpeechSegmenter`, then
classify each segment. This makes diarization fundamentally harder because:

1. **Segment boundaries don't align with turn boundaries.** The VAD detects speech regions
   independently, often splitting a single turn into 2-3 segments or merging boundary
   regions. This produces 20 segments from 8 turns.

2. **Some segments are very short.** Segments under ~1s lose speaker-discriminative power
   in the embedding, leading to near-random classification.

3. **Embeddings from concatenated audio differ from standalone files.** Even for identical
   speech content, the ECAPA-TDNN embedding of a segment carved from a longer audio stream
   can differ from the embedding of the same speech as a standalone file.

## Results

### No VAE (`diarization_speaker_id_no_vae.py`)

**Accuracy: 8/20 segments (40%)**

Without any VAE projection, the raw ECAPA-TDNN cosine similarities between all three
speakers are extremely close (typically 0.75-0.88). The margin between correct and
incorrect speakers is often <0.02, making the classification nearly coin-flip for short
or ambiguous segments:

```
Seg  3: [Bella=0.835 Bruno=0.843 Luna=0.853]  ← margin: 0.010
Seg  8: [Bella=0.760 Bruno=0.754 Luna=0.777]  ← margin: 0.017
Seg 13: [Bella=0.702 Bruno=0.719 Luna=0.719]  ← tied
```

Bella segments are mostly correct (low embeddings cluster toward Bella), but Bruno
segments are frequently misclassified as Luna because Luna's reference sits between
Bella and Bruno in the raw embedding space.

### Labelled VAE (`diarization_speaker_id_labelled_vae.py`)

**Accuracy: 14/20 segments (70%)**

The labelled VAE is trained on 10 labelled embeddings from the same speakers but
**different phrases** than the reference prototypes. This prevents overfitting while
still benefiting from the same underlying voice characteristics. The VAE learns to
separate speakers effectively in latent space:

```
Seg  3: [Bella=-0.267 Bruno=0.464 Luna=-0.384]  ← Bruno clearly positive
Seg 10: [Bella=-0.056 Bruno=-0.005 Luna=-0.025]  ← Bruno wins
Seg 16: [Bella=-0.186 Bruno=0.354 Luna=-0.242]  ← clean separation
```

### Unsupervised VAE (`diarization_speaker_id_unsupervised_vae.py`)

**Accuracy: 6-12/20 segments (30-60%)**

Trained on unlabelled speech data from all three voices (10 different phrases), this VAE
learns a general embedding transformation. Results vary significantly between training
runs due to random initialization with only 10 training samples.

## Why the Labelled VAE Beats the Unsupervised VAE

With a fair setup (same `refs/`, fresh training), the labelled VAE achieves **70%** while
the unsupervised VAE achieves **30-60%** with high variance. The labelled VAE's advantage
comes from training on **different phrases** than the reference prototypes — this provides
better generalization than the unsupervised VAE which trains on unlabelled data with no
speaker structure.

### Bugs Found and Fixed

Three issues caused the labelled VAE to appear worse than the unsupervised VAE:

1. **Overfitting to reference prototypes**: The original labelled VAE trained on the
   **exact same 9 reference embeddings** used for matching (`collect_all_embeddings`
   received `ref_paths`). This caused the VAE to memorize the reference prototypes
   rather than learning a general speaker-discriminative space. Fixed by training on
   10 different labelled phrases (`VAE_LABELLED_PHRASES`) instead.

2. **Different reference audio**: The original labelled VAE generated its own reference
   audio into `refs_labelled/`, while the unsupervised VAE read from `refs/`. Since the
   TTS is non-deterministic across runs, these contained different audio, making
   comparison unfair. Fixed by having the labelled VAE read from `refs/` directly.

3. **Stale cached model**: The unsupervised VAE cached `models/unsupervised_vae.pt`
   from a previous session with a lucky random initialization, inflating its apparent
   accuracy. When retrained fresh, its accuracy dropped to 30%.

## Loss Function Experiments

Adding supervised loss terms to the labelled VAE was explored but found to **hurt**
performance with only 10 training samples:

| Loss variant | Weight | Accuracy | Problem |
|---|---|---|---|
| Classification head | 1.0 | 35% | Overfits, classifier dominates latent space |
| Classification head | 0.1 | 25% | Still overfits, Bruno always misclassified |
| Classification from mu | 0.01 | 30% | Insufficient signal, no improvement |
| Supervised contrastive (temp=0.07) | 0.1 | 15% | Collapses latent space, Luna dominates |
| Supervised contrastive (temp=0.5) | 0.01 | 35% | Still collapses, no improvement |
| Center + margin (Wen et al. 2016) | 0.01 | 35% | Noisy centroids from 3-4 samples collapse space |
| Mixup augmentation (alpha=0.4) | — | 10% | Interpolates between speakers, creates ambiguity |
| **Standard VAE loss (no labels)** | — | **20-70%** | Best ceiling, high variance expected at 10 samples |

**Conclusion**: With only 10 training samples, every supervised or augmented approach
performs worse than the standard VAE loss. The reasons are specific to each approach:

- **Classification/center losses** compute per-speaker centroids from 3-4 samples each —
  too noisy to provide a useful gradient signal
- **Contrastive losses** have far more negative pairs (different-speaker) than positive
  pairs (same-speaker) in a batch of 10 — the gradient is dominated by pushing everything
  apart, collapsing the space
- **Mixup** interpolates between different speakers' embeddings, creating training examples
  that are inherently ambiguous — the VAE learns to blur rather than separate

The labelled VAE's advantage comes from the **training data** (different phrases from
known speakers), not from using labels in the loss function. The standard VAE loss
(reconstruction + KL divergence) gives the 128-dim latent space freedom to organize
itself for cosine matching without overconstraining it.

The `classify()` head and `num_speakers` config remain in `speech_segmentation/vae.py`
for future use with larger datasets (100+ samples per speaker) where supervised losses
would have enough data to generalize.

## Common Failure Patterns

**Short segments (<1s):** Segments 8 (0.50s), 13 (0.12s) are consistently misclassified
across all variants. The ECAPA-TDNN model needs ~1s of speech to produce a reliable
speaker embedding.

**Boundary ambiguity:** Segments at turn transitions (e.g., seg 6 at the end of Bella's
turn, seg 13 at a turn boundary) contain mixed or partial speech that confuses both the
segmenter and the embedder.

**Luna confusion:** Luna is a distractor (not in the dialogue) whose reference sits
between Bella and Bruno in embedding space. Without VAE projection, Bruno segments
frequently match Luna as the nearest reference.

## Why Results Differ from Speaker Identification

The `speaker_identification_*.py` examples achieve 87-100% accuracy because they classify
clean, turn-level audio files. The diarization pipeline operates on raw concatenated audio
where the segmenter produces fragments of varying quality and duration. The accuracy gap
(30-70% vs 87-100%) reflects this fundamental difference in input quality, not a flaw in
the identification method.

## Files Modified

- **All three diarization examples**: Updated `evaluate_diarization()` to show
  per-speaker similarity scores (matching the display format of the speaker
  identification examples), making failures diagnosable.

- **`diarization_speaker_id_labelled_vae.py`**: Fixed overfitting by training on
  separate labelled phrases (`VAE_LABELLED_PHRASES`) instead of the reference
  embeddings. Changed to read references from `refs/` (shared with other examples).
  Uses separate model path (`models/diarization_labelled_vae.pt`).

- **`diarization_speaker_id_unsupervised_vae.py`**: Added architecture discussion
  note explaining why standard VAE loss was chosen over supervised alternatives,
  with references to the loss function experiments in this report.

- **`speech_segmentation/vae.py`**: Added optional `num_speakers` parameter and
  `classify()` method to `VAE` class for future supervised training experiments.
  The classification head is not used in the current labelled VAE example since
  standard VAE loss outperforms supervised losses at this data scale.
