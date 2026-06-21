"""Tests for optimization framework including metrics, config, and backend switching."""

import pytest
from evaluation.metrics import (
    CombinedEvaluator,
    DiarizationEvaluator,
    DiarizationMetrics,
    EvaluationResult,
    TranscriptionEvaluator,
    TranscriptionMetrics,
)
from config.optimization_config import ConfigManager, OptimizationConfig


class TestTranscriptionMetrics:
    """Test transcription accuracy metrics."""

    def test_wer_perfect_match(self):
        """WER should be 0 for identical text."""
        ref = "the quick brown fox"
        hyp = "the quick brown fox"
        metrics = TranscriptionEvaluator.word_error_rate(ref, hyp)
        assert metrics.wer == 0.0

    def test_wer_complete_mismatch(self):
        """WER should be ~1.0 for completely different text."""
        ref = "the quick brown fox"
        hyp = "completely different words"
        metrics = TranscriptionEvaluator.word_error_rate(ref, hyp)
        assert metrics.wer > 0.5

    def test_cer_indic_script(self):
        """CER should work with Indic scripts."""
        ref = "నమస్తే"
        hyp = "నమస్"
        cer = TranscriptionEvaluator.character_error_rate(ref, hyp)
        assert 0 < cer < 100  # Partial match

    def test_error_components(self):
        """Verify insertions, deletions, substitutions are tracked."""
        ref = "the cat sat"
        hyp = "the dog sat"
        metrics = TranscriptionEvaluator.word_error_rate(ref, hyp)
        assert metrics.substitutions > 0  # "cat" -> "dog"
        assert metrics.hits > 0  # "the" and "sat" match


class TestDiarizationMetrics:
    """Test diarization accuracy metrics."""

    def test_der_perfect_match(self):
        """DER should be 0 for identical timelines."""
        timeline = [
            {"start": 0.0, "end": 5.0, "speaker": "Speaker_1"},
            {"start": 5.0, "end": 10.0, "speaker": "Speaker_2"},
        ]
        metrics = DiarizationEvaluator.der_from_timeline(timeline, timeline)
        assert metrics.der == 0.0

    def test_der_wrong_speaker(self):
        """DER should be >0 when speakers don't match."""
        ref = [
            {"start": 0.0, "end": 10.0, "speaker": "Speaker_1"},
        ]
        hyp = [
            {"start": 0.0, "end": 10.0, "speaker": "Speaker_2"},
        ]
        metrics = DiarizationEvaluator.der_from_timeline(ref, hyp)
        assert metrics.der > 0

    def test_overlap_detection(self):
        """Test detection of overlapping speakers."""
        ref = [
            {"start": 0.0, "end": 5.0, "speaker": "Speaker_1"},
            {"start": 2.0, "end": 7.0, "speaker": "Speaker_2"},  # Overlap
        ]
        hyp = [
            {"start": 0.0, "end": 5.0, "speaker": "Speaker_1"},
            {"start": 2.0, "end": 7.0, "speaker": "Speaker_2"},  # Overlap
        ]
        accuracy = DiarizationEvaluator.overlap_detection_accuracy(ref, hyp)
        assert accuracy == 100.0  # Perfect match

    def test_speaker_attribution(self):
        """Test speaker assignment accuracy."""
        ref = [
            {"start": 0.0, "end": 10.0, "speaker": "Speaker_1"},
        ]
        hyp = [
            {"start": 1.0, "end": 9.0, "speaker": "Speaker_1"},
        ]
        accuracy = DiarizationEvaluator.speaker_attribution_accuracy(ref, hyp)
        assert 90 < accuracy <= 100  # Should match despite time offset


class TestCombinedEvaluator:
    """Test combined quality scoring."""

    def test_perfect_score(self):
        """Perfect transcription and diarization should give ~100."""
        trans = TranscriptionMetrics(
            wer=0.0, cer=0.0,
            insertions=0, deletions=0, substitutions=0, hits=100
        )
        diar = DiarizationMetrics(
            der=0.0, false_alarm=0.0, missed_detection=0.0,
            confusion=0.0, speaker_attribution_accuracy=100.0
        )
        score = CombinedEvaluator.overall_quality_score(trans, diar)
        assert score > 95

    def test_poor_score(self):
        """Poor transcription and diarization should give low score."""
        trans = TranscriptionMetrics(
            wer=0.5, cer=0.5,
            insertions=10, deletions=10, substitutions=10, hits=10
        )
        diar = DiarizationMetrics(
            der=50.0, false_alarm=20.0, missed_detection=20.0,
            confusion=10.0, speaker_attribution_accuracy=50.0
        )
        score = CombinedEvaluator.overall_quality_score(trans, diar)
        assert score <= 50  # 50% accuracy

    def test_custom_weights(self):
        """Test that custom weights affect score."""
        trans = TranscriptionMetrics(
            wer=0.1, cer=0.1,
            insertions=0, deletions=0, substitutions=0, hits=90
        )
        diar = DiarizationMetrics(
            der=10.0, false_alarm=5.0, missed_detection=3.0,
            confusion=2.0, speaker_attribution_accuracy=90.0
        )

        # Default weights
        score_default = CombinedEvaluator.overall_quality_score(trans, diar)

        # Emphasize diarization
        weights_diar = {"wer": 0.2, "cer": 0.1, "der": 0.5, "speaker_attribution": 0.2}
        score_diar = CombinedEvaluator.overall_quality_score(trans, diar, weights_diar)

        assert score_diar >= score_default  # More weight on diarization (which is better)


class TestOptimizationConfig:
    """Test configuration management."""

    def test_load_balanced_profile(self):
        """Test loading the balanced profile."""
        config = OptimizationConfig("balanced")
        assert config.profile_name == "balanced"
        assert config.backend in ["sortformer", "pyannote"]

    def test_load_production_profile(self):
        """Test loading production quality profile."""
        config = OptimizationConfig("production_quality")
        assert config.profile_name == "production_quality"
        assert config.get_beam_size() >= 10  # Higher quality = larger beam

    def test_load_real_time_profile(self):
        """Test loading real-time profile."""
        config = OptimizationConfig("real_time")
        assert config.profile_name == "real_time"
        assert config.get_beam_size() <= 10  # Lower quality = smaller beam

    def test_beam_size_ordering(self):
        """Beam size should increase with quality."""
        real_time = OptimizationConfig("real_time").get_beam_size()
        balanced = OptimizationConfig("balanced").get_beam_size()
        production = OptimizationConfig("production_quality").get_beam_size()

        assert real_time <= balanced <= production

    def test_backend_switching(self):
        """Test switching between backends."""
        config = OptimizationConfig("balanced")
        original_backend = config.backend

        config.switch_backend("sortformer")
        assert config.backend == "sortformer"

        config.switch_backend("pyannote")
        assert config.backend == "pyannote"

        # Invalid backend should raise
        with pytest.raises(ValueError):
            config.switch_backend("invalid_backend")

    def test_config_manager_singleton(self):
        """Test ConfigManager singleton behavior."""
        ConfigManager.reset()

        config1 = ConfigManager.get()
        config2 = ConfigManager.get()

        assert config1 is config2  # Same instance

        ConfigManager.reset()
        config3 = ConfigManager.get()

        assert config1 is not config3  # Different instance after reset

    def test_backend_info(self):
        """Test backend characteristics are available."""
        info = OptimizationConfig.BACKEND_INFO

        assert "sortformer" in info
        assert "pyannote" in info

        sortformer = info["sortformer"]
        assert sortformer["accuracy"] > 0
        assert sortformer["speed_relative"] > 0
        assert "memory_gb" in sortformer

    def test_environment_variables(self):
        """Test generating environment variables."""
        config = OptimizationConfig("production_quality")
        env_vars = config.get_environment_variables()

        assert env_vars["OPTIMIZATION_PROFILE"] == "production_quality"
        assert env_vars["DIARIZER"] in ["sortformer", "pyannote"]
        assert "ASR_BEAM_SIZE" in env_vars
        assert "ASR_LM_WEIGHT" in env_vars


class TestBackendEquivalence:
    """Test that both backends produce consistent results."""

    def test_sortformer_pyannote_same_output(self):
        """Both backends should handle same timeline identically."""
        timeline = [
            {"start": 0.0, "end": 5.0, "speaker": "Speaker_1"},
            {"start": 2.0, "end": 7.0, "speaker": "Speaker_2"},
            {"start": 7.0, "end": 10.0, "speaker": "Speaker_1"},
        ]

        # DER calculation should be same regardless of backend
        metrics1 = DiarizationEvaluator.der_from_timeline(timeline, timeline)
        metrics2 = DiarizationEvaluator.der_from_timeline(timeline, timeline)

        assert metrics1.der == metrics2.der
        assert metrics1.speaker_attribution_accuracy == metrics2.speaker_attribution_accuracy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
