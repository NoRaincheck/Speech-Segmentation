"""Speech segmentation and speaker identification library.

Core inference for:
- Speech segmentation (detecting who speaks when)
- Speaker identification using 1-shot or few-shot learning
"""

from speech_segmentation.diarizer import Diarizer
from speech_segmentation.embedding import SpeakerEmbedder
from speech_segmentation.segmentation import SpeechSegmenter

__all__ = ["Diarizer", "SpeakerEmbedder", "SpeechSegmenter"]
