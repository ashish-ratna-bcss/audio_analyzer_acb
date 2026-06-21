"""
Verbosity-specific metrics for measuring word retention and filler word capture.

Complements WER/CER with metrics focused on preserving all spoken content.
"""

from dataclasses import dataclass
from typing import Set


@dataclass
class VerbosityMetrics:
    """Verbosity-focused metrics."""
    word_count_hypothesis: int      # Total words in hypothesis
    word_count_reference: int       # Total words in reference
    word_retention_rate: float      # % of reference words captured (0-1)
    filler_word_count_ref: int      # Filler words in reference
    filler_word_count_hyp: int      # Filler words in hypothesis
    filler_word_recall: float       # % of filler words captured (0-1)
    hesitation_count_ref: int       # False starts/repairs in reference
    hesitation_count_hyp: int       # False starts/repairs in hypothesis
    average_confidence: float       # Average token confidence


class VerbosityEvaluator:
    """Evaluate transcription verbosity and word retention."""

    FILLER_WORDS = {
        "um", "uh", "ah", "er", "hmm", "huh", "uh-huh",
        "you know", "like", "basically", "literally",
        "actually", "really", "honestly", "i mean",
        "sort of", "kind of", "so", "well", "you see"
    }

    HESITATION_MARKERS = {
        "er", "um", "uh", "ah",  # Filled pauses
        "[disfluency]", "[false start]", "[repair]"
    }

    @staticmethod
    def word_retention_rate(
        reference: str,
        hypothesis: str
    ) -> float:
        """Calculate % of reference words retained in hypothesis.

        Measures how many words from the ground truth appear in the output.
        Higher = more complete transcription (less aggressive cleaning).

        Args:
            reference: Ground truth transcript
            hypothesis: Generated transcript

        Returns:
            Retention rate (0.0 = no words retained, 1.0 = all words retained)
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        if not ref_words:
            return 1.0 if not hyp_words else 0.0

        # Count unique words from reference that appear in hypothesis
        ref_word_set = set(ref_words)
        hyp_word_set = set(hyp_words)

        retained = len(ref_word_set & hyp_word_set)
        return retained / len(ref_word_set)

    @staticmethod
    def filler_word_count(
        transcript: str,
        filler_words: Set[str] = None
    ) -> int:
        """Count filler words in transcript.

        Args:
            transcript: Text to analyze
            filler_words: Custom filler word set (default: predefined set)

        Returns:
            Number of filler words found
        """
        if filler_words is None:
            filler_words = VerbosityEvaluator.FILLER_WORDS

        words = transcript.lower().split()
        count = sum(1 for word in words if word in filler_words)
        return count

    @staticmethod
    def filler_word_recall(
        reference: str,
        hypothesis: str,
        filler_words: Set[str] = None
    ) -> float:
        """Calculate % of filler words from reference captured in hypothesis.

        Important for court proceedings where natural speech patterns matter.

        Args:
            reference: Ground truth
            hypothesis: Generated transcript
            filler_words: Custom filler word set

        Returns:
            Recall rate (0.0-1.0)
        """
        if filler_words is None:
            filler_words = VerbosityEvaluator.FILLER_WORDS

        ref_fillers = VerbosityEvaluator.filler_word_count(
            reference, filler_words
        )
        hyp_fillers = VerbosityEvaluator.filler_word_count(
            hypothesis, filler_words
        )

        if ref_fillers == 0:
            return 1.0 if hyp_fillers == 0 else 0.0

        return min(hyp_fillers / ref_fillers, 1.0)

    @staticmethod
    def calculate_verbosity_metrics(
        reference: str,
        hypothesis: str,
        confidence_scores: list = None
    ) -> VerbosityMetrics:
        """Calculate all verbosity metrics.

        Args:
            reference: Ground truth transcript
            hypothesis: Generated transcript
            confidence_scores: Token-level confidence scores (optional)

        Returns:
            VerbosityMetrics with all measurements
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        word_retention = VerbosityEvaluator.word_retention_rate(
            reference, hypothesis
        )

        ref_fillers = VerbosityEvaluator.filler_word_count(reference)
        hyp_fillers = VerbosityEvaluator.filler_word_count(hypothesis)

        filler_recall = VerbosityEvaluator.filler_word_recall(
            reference, hypothesis
        )

        # Average confidence (default to 0.5 if not provided)
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores and len(confidence_scores) > 0
            else 0.5
        )

        return VerbosityMetrics(
            word_count_hypothesis=len(hyp_words),
            word_count_reference=len(ref_words),
            word_retention_rate=word_retention,
            filler_word_count_ref=ref_fillers,
            filler_word_count_hyp=hyp_fillers,
            filler_word_recall=filler_recall,
            hesitation_count_ref=0,  # Placeholder
            hesitation_count_hyp=0,  # Placeholder
            average_confidence=avg_confidence,
        )

    @staticmethod
    def verbosity_summary(metrics: VerbosityMetrics) -> str:
        """Generate human-readable verbosity summary."""
        return f"""
VERBOSITY METRICS:
  Word Retention:     {metrics.word_retention_rate:.1%}
    ({metrics.word_count_hypothesis}/{metrics.word_count_reference} words)

  Filler Words:       {metrics.filler_word_recall:.1%}
    ({metrics.filler_word_count_hyp}/{metrics.filler_word_count_ref} fillers)

  Average Confidence: {metrics.average_confidence:.3f}
"""


class VerboseTranscriptOutput:
    """Transcript with token-level details for verbosity analysis."""

    def __init__(
        self,
        transcript: str,
        tokens: list = None,
        confidence_scores: list = None,
        timestamps: list = None
    ):
        """Initialize verbose transcript.

        Args:
            transcript: Full transcript text
            tokens: List of individual tokens/words
            confidence_scores: Confidence for each token
            timestamps: Start/end times for each token
        """
        self.transcript = transcript
        self.tokens = tokens or transcript.split()
        self.confidence_scores = confidence_scores or [0.5] * len(self.tokens)
        self.timestamps = timestamps or [(0, 0)] * len(self.tokens)

    @property
    def word_count(self) -> int:
        """Total word count."""
        return len(self.tokens)

    @property
    def average_confidence(self) -> float:
        """Average confidence across all tokens."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)

    @property
    def low_confidence_tokens(self, threshold: float = 0.5) -> list:
        """Get tokens with confidence below threshold."""
        return [
            (token, conf)
            for token, conf in zip(self.tokens, self.confidence_scores)
            if conf < threshold
        ]

    def to_markdown_verbose(self) -> str:
        """Export as markdown with confidence markers and timing.

        Format:
        word1 [00.0-00.5, conf:0.92] word2 [00.5-01.0, conf:0.45][LOW]
        """
        output = []

        for token, confidence, (start, end) in zip(
            self.tokens, self.confidence_scores, self.timestamps
        ):
            # Add confidence marker
            if confidence < 0.5:
                marker = " [LOW-CONF]"
            elif confidence < 0.7:
                marker = " [MED-CONF]"
            else:
                marker = ""

            # Add timing
            timing = f" [{start:.2f}-{end:.2f}s, {confidence:.2f}]"

            output.append(f"{token}{timing}{marker}")

        return " ".join(output)

    def to_json(self) -> dict:
        """Export as JSON with token details."""
        return {
            "transcript": self.transcript,
            "word_count": self.word_count,
            "average_confidence": self.average_confidence,
            "tokens": [
                {
                    "word": token,
                    "confidence": conf,
                    "start_time": start,
                    "end_time": end,
                }
                for token, conf, (start, end) in zip(
                    self.tokens, self.confidence_scores, self.timestamps
                )
            ],
        }
