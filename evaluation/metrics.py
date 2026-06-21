"""
Evaluation metrics for transcription and diarization accuracy.

Implements:
- WER (Word Error Rate): Transcription word-level accuracy
- CER (Character Error Rate): Character-level accuracy (important for Indic scripts)
- DER (Diarization Error Rate): Speaker segmentation accuracy
- Speaker Attribution Accuracy: Correct speaker assignment rate
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jiwer
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionMetrics:
    """Transcription accuracy metrics."""
    wer: float
    cer: float
    insertions: int
    deletions: int
    substitutions: int
    hits: int


@dataclass
class DiarizationMetrics:
    """Diarization accuracy metrics."""
    der: float  # Diarization Error Rate (%)
    false_alarm: float  # Extra speaker time (%)
    missed_detection: float  # Missed speaker time (%)
    confusion: float  # Speaker confusion (%)
    speaker_attribution_accuracy: float  # % correct speaker assignment


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    file_id: str
    transcription: TranscriptionMetrics
    diarization: DiarizationMetrics
    overall_score: float  # Weighted combination


class TranscriptionEvaluator:
    """Evaluate transcription accuracy against reference."""

    @staticmethod
    def word_error_rate(reference: str, hypothesis: str) -> TranscriptionMetrics:
        """Calculate WER (Word Error Rate).

        Args:
            reference: Ground truth transcript
            hypothesis: Predicted transcript

        Returns:
            TranscriptionMetrics with WER, CER, edit distances
        """
        # Normalize text (lowercase, remove extra spaces)
        ref_text = reference.lower().strip()
        hyp_text = hypothesis.lower().strip()

        # Split into words
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()

        # Calculate metrics using jiwer
        wer = jiwer.wer(ref_text, hyp_text)
        cer = jiwer.cer(ref_text, hyp_text)

        # Calculate individual error types
        output = jiwer.process_characters(ref_text, hyp_text)

        return TranscriptionMetrics(
            wer=wer,
            cer=cer,
            insertions=output.insertions,
            deletions=output.deletions,
            substitutions=output.substitutions,
            hits=output.hits,
        )

    @staticmethod
    def character_error_rate(reference: str, hypothesis: str) -> float:
        """Calculate CER - critical for Indic scripts.

        Returns CER as percentage (0-100).
        """
        return jiwer.cer(reference.lower(), hypothesis.lower()) * 100


class DiarizationEvaluator:
    """Evaluate diarization (speaker segmentation) accuracy."""

    FRAME_RATE = 100  # 10ms frames

    @staticmethod
    def der_from_timeline(
        reference_timeline: List[Dict],
        hypothesis_timeline: List[Dict],
        step: float = 0.010,  # 10ms steps
    ) -> DiarizationMetrics:
        """Calculate Diarization Error Rate (DER) using frame-level comparison.

        Args:
            reference_timeline: Ground truth speaker timeline
                [{start, end, speaker}, ...]
            hypothesis_timeline: Predicted speaker timeline
            step: Frame duration in seconds (default 10ms)

        Returns:
            DiarizationMetrics including DER and components
        """
        if not reference_timeline or not hypothesis_timeline:
            logger.warning("Empty timeline provided")
            return DiarizationMetrics(der=100.0, false_alarm=0, missed_detection=0,
                                     confusion=0, speaker_attribution_accuracy=0)

        # Get overall duration
        ref_end = max(s["end"] for s in reference_timeline)
        hyp_end = max(s["end"] for s in hypothesis_timeline)
        duration = max(ref_end, hyp_end)

        # Create frame-level speaker assignments
        num_frames = int(duration / step) + 1
        ref_frames = DiarizationEvaluator._timeline_to_frames(
            reference_timeline, num_frames, step
        )
        hyp_frames = DiarizationEvaluator._timeline_to_frames(
            hypothesis_timeline, num_frames, step
        )

        # Calculate error components
        false_alarm = 0  # Hypothesis has speaker, reference doesn't
        missed_detection = 0  # Reference has speaker, hypothesis doesn't
        confusion = 0  # Both have speakers but different ones

        for ref_speakers, hyp_speakers in zip(ref_frames, hyp_frames):
            ref_set = set(ref_speakers)
            hyp_set = set(hyp_speakers)

            if not ref_set and hyp_set:
                false_alarm += 1
            elif ref_set and not hyp_set:
                missed_detection += 1
            elif ref_set != hyp_set:
                confusion += 1

        # Normalize by total frames
        total_frames = len(ref_frames)
        false_alarm_rate = (false_alarm / total_frames * 100) if total_frames else 0
        missed_detection_rate = (missed_detection / total_frames * 100) if total_frames else 0
        confusion_rate = (confusion / total_frames * 100) if total_frames else 0

        der = false_alarm_rate + missed_detection_rate + confusion_rate

        return DiarizationMetrics(
            der=der,
            false_alarm=false_alarm_rate,
            missed_detection=missed_detection_rate,
            confusion=confusion_rate,
            speaker_attribution_accuracy=100 - der,  # Inverse of error
        )

    @staticmethod
    def _timeline_to_frames(
        timeline: List[Dict], num_frames: int, step: float
    ) -> List[List[str]]:
        """Convert timeline to frame-level speaker assignments."""
        frames = [[] for _ in range(num_frames)]

        for segment in timeline:
            start_frame = int(segment["start"] / step)
            end_frame = int(segment["end"] / step)
            speaker = segment["speaker"]

            for frame_idx in range(start_frame, min(end_frame + 1, num_frames)):
                if speaker not in frames[frame_idx]:
                    frames[frame_idx].append(speaker)

        return frames

    @staticmethod
    def speaker_attribution_accuracy(
        reference_timeline: List[Dict], hypothesis_timeline: List[Dict]
    ) -> float:
        """Calculate % of segments with correct speaker assignment.

        Assigns speakers to hypothesis segments by finding closest reference
        speaker based on time overlap.
        """
        if not reference_timeline or not hypothesis_timeline:
            return 0.0

        correct = 0

        for hyp_seg in hypothesis_timeline:
            # Find best matching reference segment (max overlap)
            best_overlap = 0
            best_speaker = None

            for ref_seg in reference_timeline:
                overlap_start = max(hyp_seg["start"], ref_seg["start"])
                overlap_end = min(hyp_seg["end"], ref_seg["end"])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = ref_seg["speaker"]

            if best_speaker and hyp_seg["speaker"] == best_speaker:
                correct += 1

        return (correct / len(hypothesis_timeline)) * 100 if hypothesis_timeline else 0

    @staticmethod
    def overlap_detection_accuracy(
        reference_timeline: List[Dict], hypothesis_timeline: List[Dict]
    ) -> float:
        """Calculate accuracy in detecting overlapping speakers.

        Returns % of overlapping time windows correctly identified.
        """
        # Find overlapping time windows in reference
        ref_overlaps = DiarizationEvaluator._find_overlaps(reference_timeline)
        hyp_overlaps = DiarizationEvaluator._find_overlaps(hypothesis_timeline)

        if not ref_overlaps:
            return 100.0 if not hyp_overlaps else 50.0

        # Calculate recall and precision
        detected = len(ref_overlaps & hyp_overlaps)
        recall = detected / len(ref_overlaps) if ref_overlaps else 0
        precision = detected / len(hyp_overlaps) if hyp_overlaps else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1 * 100

    @staticmethod
    def _find_overlaps(timeline: List[Dict], step: float = 0.010) -> set:
        """Find time windows with overlapping speakers (multiple speakers at once)."""
        overlaps = set()
        duration = max(s["end"] for s in timeline) if timeline else 0
        num_frames = int(duration / step) + 1

        for frame_idx in range(num_frames):
            frame_time = frame_idx * step
            speakers = set()

            for segment in timeline:
                if segment["start"] <= frame_time < segment["end"]:
                    speakers.add(segment["speaker"])

            if len(speakers) > 1:
                overlaps.add(frame_idx)

        return overlaps


class CombinedEvaluator:
    """Combine transcription and diarization metrics into overall score."""

    # Weights for combined scoring (sum = 1.0)
    WEIGHTS = {
        "wer": 0.40,  # Transcription accuracy (primary)
        "cer": 0.20,  # Character accuracy (Indic scripts)
        "der": 0.25,  # Diarization accuracy
        "speaker_attribution": 0.15,  # Speaker assignment accuracy
    }

    @staticmethod
    def overall_quality_score(
        transcription: TranscriptionMetrics,
        diarization: DiarizationMetrics,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculate combined quality score (0-100).

        Args:
            transcription: WER/CER metrics
            diarization: DER and speaker attribution
            weights: Custom weight distribution

        Returns:
            Overall score where 100 = perfect, 0 = completely wrong
        """
        w = weights or CombinedEvaluator.WEIGHTS

        # Convert error rates to accuracy (100 - error_rate)
        wer_score = 100 - (transcription.wer * 100)
        cer_score = 100 - (transcription.cer * 100)
        der_score = diarization.speaker_attribution_accuracy
        attr_score = diarization.speaker_attribution_accuracy

        # Clamp to [0, 100]
        wer_score = max(0, min(100, wer_score))
        cer_score = max(0, min(100, cer_score))
        der_score = max(0, min(100, der_score))
        attr_score = max(0, min(100, attr_score))

        combined = (
            w["wer"] * wer_score
            + w["cer"] * cer_score
            + w["der"] * der_score
            + w["speaker_attribution"] * attr_score
        )

        return combined


class EvaluationReport:
    """Generate evaluation reports and comparisons."""

    @staticmethod
    def to_json(result: EvaluationResult) -> Dict:
        """Convert result to JSON-serializable dict."""
        return {
            "file_id": result.file_id,
            "transcription": {
                "wer": result.transcription.wer,
                "cer": result.transcription.cer,
                "insertions": result.transcription.insertions,
                "deletions": result.transcription.deletions,
                "substitutions": result.transcription.substitutions,
                "hits": result.transcription.hits,
            },
            "diarization": {
                "der": result.diarization.der,
                "false_alarm": result.diarization.false_alarm,
                "missed_detection": result.diarization.missed_detection,
                "confusion": result.diarization.confusion,
                "speaker_attribution": result.diarization.speaker_attribution_accuracy,
            },
            "overall_score": result.overall_score,
        }

    @staticmethod
    def save(result: EvaluationResult, path: Path) -> None:
        """Save evaluation result to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(EvaluationReport.to_json(result), f, indent=2)

    @staticmethod
    def print_summary(result: EvaluationResult) -> str:
        """Print human-readable summary."""
        summary = f"""
=== Evaluation Report: {result.file_id} ===

TRANSCRIPTION:
  WER (Word Error Rate):        {result.transcription.wer:.1%}
  CER (Character Error Rate):   {result.transcription.cer:.1%}
  Hits: {result.transcription.hits} | Subs: {result.transcription.substitutions} | Del: {result.transcription.deletions} | Ins: {result.transcription.insertions}

DIARIZATION:
  DER (Diarization Error Rate): {result.diarization.der:.2f}%
    - False Alarm (extra speaker):     {result.diarization.false_alarm:.2f}%
    - Missed Detection (missing speaker): {result.diarization.missed_detection:.2f}%
    - Confusion (wrong speaker):       {result.diarization.confusion:.2f}%
  Speaker Attribution Accuracy: {result.diarization.speaker_attribution_accuracy:.1f}%

OVERALL SCORE: {result.overall_score:.1f}/100
"""
        return summary.strip()
