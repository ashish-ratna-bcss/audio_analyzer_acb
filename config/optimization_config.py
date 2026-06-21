"""
Configuration management for optimization profiles and backend selection.

Allows dynamic selection of:
- Quality profiles (production_quality, balanced, real_time)
- Diarization backends (sortformer, pyannote)
- Hyperparameters for each component
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class OptimizationConfig:
    """Load and manage optimization profiles."""

    PROFILES_DIR = Path(__file__).parent / "profiles"
    DEFAULT_PROFILE = "balanced"

    # Backend characteristics for reference
    BACKEND_INFO = {
        "sortformer": {
            "description": "NVIDIA Sortformer (isolated NeMo sidecar)",
            "accuracy": 0.92,  # Empirical DER
            "speed_relative": 0.8,  # Slower than pyannote
            "overlap_handling": "native",  # Handles overlaps natively
            "memory_gb": 2,
            "dependencies": ["nemo_toolkit"],
        },
        "pyannote": {
            "description": "PyAnnote Audio (lightweight)",
            "accuracy": 0.88,  # Empirical DER
            "speed_relative": 1.0,  # Baseline
            "overlap_handling": "post_process",  # Post-processing needed
            "memory_gb": 1.5,
            "dependencies": ["pyannote.audio"],
        },
    }

    def __init__(self, profile_name: Optional[str] = None):
        """Initialize configuration from profile.

        Args:
            profile_name: Name of profile (production_quality, balanced, real_time)
                         or None to use env var or default
        """
        self.profile_name = (
            profile_name
            or os.getenv("OPTIMIZATION_PROFILE", self.DEFAULT_PROFILE)
        )
        self.config = self._load_profile(self.profile_name)
        self.profile_path = self.PROFILES_DIR / f"{self.profile_name}.yaml"

        logger.info(
            f"Loaded optimization profile: {self.profile_name} "
            f"({self.config.get('description', 'N/A')})"
        )

    def _load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Load YAML profile from disk."""
        profile_path = self.PROFILES_DIR / f"{profile_name}.yaml"

        if not profile_path.exists():
            logger.warning(f"Profile not found: {profile_path}, using defaults")
            return {"profile_name": profile_name, "description": "Default profile"}

        with open(profile_path) as f:
            return yaml.safe_load(f)

    @property
    def backend(self) -> str:
        """Get selected diarization backend."""
        backend = self.config.get("diarization", {}).get("backend", "pyannote")
        return backend

    @property
    def transcription_config(self) -> Dict[str, Any]:
        """Get transcription parameters."""
        return self.config.get("transcription", {})

    @property
    def diarization_config(self) -> Dict[str, Any]:
        """Get diarization parameters."""
        return self.config.get("diarization", {})

    @property
    def backend_config(self) -> Dict[str, Any]:
        """Get configuration for selected backend."""
        backend = self.backend
        return self.diarization_config.get(backend, {})

    @property
    def postprocessing_config(self) -> Dict[str, Any]:
        """Get post-processing parameters."""
        return self.config.get("postprocessing", {})

    @property
    def expected_metrics(self) -> Dict[str, float]:
        """Get expected performance metrics for this profile."""
        return self.config.get("expected_metrics", {})

    def get_beam_size(self) -> int:
        """Get beam size for transcription."""
        return self.transcription_config.get("decoding", {}).get("beam_size", 10)

    def get_lm_weight(self) -> float:
        """Get language model weight."""
        return self.transcription_config.get("decoding", {}).get("lm_weight", 1.0)

    def get_vad_threshold(self) -> float:
        """Get VAD threshold."""
        return self.transcription_config.get("preprocessing", {}).get("vad_threshold", 0.5)

    def get_min_segment_duration(self) -> float:
        """Get minimum segment duration."""
        return self.transcription_config.get("segmentation", {}).get(
            "min_segment_duration", 1.0
        )

    def get_sortformer_params(self) -> Dict[str, Any]:
        """Get Sortformer-specific parameters."""
        return self.diarization_config.get("sortformer", {})

    def get_pyannote_params(self) -> Dict[str, Any]:
        """Get PyAnnote-specific parameters."""
        return self.diarization_config.get("pyannote", {})

    def switch_backend(self, backend_name: str) -> None:
        """Switch to different diarization backend at runtime.

        Args:
            backend_name: "sortformer" or "pyannote"

        Raises:
            ValueError: If backend not recognized
        """
        if backend_name not in self.BACKEND_INFO:
            raise ValueError(
                f"Unknown backend: {backend_name}. "
                f"Available: {list(self.BACKEND_INFO.keys())}"
            )

        self.config["diarization"]["backend"] = backend_name
        logger.info(
            f"Switched diarization backend to {backend_name} "
            f"({self.BACKEND_INFO[backend_name]['description']})"
        )

    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables to set for this profile."""
        env_vars = {
            "OPTIMIZATION_PROFILE": self.profile_name,
            "DIARIZER": self.backend,
            "ASR_BEAM_SIZE": str(self.get_beam_size()),
            "ASR_LM_WEIGHT": str(self.get_lm_weight()),
            "VAD_THRESHOLD": str(self.get_vad_threshold()),
        }
        return env_vars

    def to_dict(self) -> Dict[str, Any]:
        """Export full configuration as dict."""
        return self.config.copy()

    def __repr__(self) -> str:
        return (
            f"OptimizationConfig(profile={self.profile_name}, "
            f"backend={self.backend}, "
            f"beam_size={self.get_beam_size()})"
        )


class ConfigManager:
    """Global configuration manager singleton."""

    _instance: Optional[OptimizationConfig] = None

    @classmethod
    def get(cls, profile: Optional[str] = None) -> OptimizationConfig:
        """Get or create global config instance."""
        if cls._instance is None:
            cls._instance = OptimizationConfig(profile)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset global config (for testing)."""
        cls._instance = None

    @classmethod
    def switch_profile(cls, profile_name: str) -> OptimizationConfig:
        """Switch to different profile."""
        cls._instance = OptimizationConfig(profile_name)
        return cls._instance

    @classmethod
    def switch_backend(cls, backend_name: str) -> None:
        """Switch diarization backend."""
        config = cls.get()
        config.switch_backend(backend_name)


# Integration with config.py
def apply_optimization_config() -> None:
    """Apply optimization config to main config module.

    Call this from config.py to load profile-based parameters.
    """
    import config as main_config

    opt_config = ConfigManager.get()

    # Set diarizer backend
    main_config.DIARIZER = opt_config.backend

    # Set transcription parameters
    beam_size = opt_config.get_beam_size()
    lm_weight = opt_config.get_lm_weight()

    logger.info(
        f"Applied optimization config: diarizer={opt_config.backend}, "
        f"beam_size={beam_size}, lm_weight={lm_weight}"
    )
