"""Unit tests for L2 audio analysis + the enhancement-gating decision."""
import importlib

import numpy as np
import soundfile as sf

import config
from services import audio_analysis_service as aa


def _wav(tmp_path, samples, sr=16000):
    p = tmp_path / "a.wav"
    sf.write(str(p), samples.astype("float32"), sr)
    return str(p)


def test_analyze_returns_fields(tmp_path):
    sig = 0.3 * np.sin(2 * np.pi * 300 * np.arange(16000) / 16000)
    prof = aa.analyze(_wav(tmp_path, sig))
    assert set(prof) == {"mean_dbfs", "snr_db", "clipping_ratio", "reverb_proxy"}
    assert prof["mean_dbfs"] is not None and prof["snr_db"] is not None


def test_analyze_missing_file_safe():
    prof = aa.analyze("/no/such/file.wav")
    assert prof == {"mean_dbfs": None, "snr_db": None,
                    "clipping_ratio": None, "reverb_proxy": None}


def test_analyze_clipping_detected(tmp_path):
    sig = np.ones(16000)            # fully clipped
    prof = aa.analyze(_wav(tmp_path, sig))
    assert prof["clipping_ratio"] > 0.9


def test_needs_enhancement_low_snr():
    # loud but noisy (low SNR) -> enhance (the ACB-sting case)
    assert aa.needs_enhancement({"snr_db": 8.0, "mean_dbfs": -23.0}) is True


def test_needs_enhancement_quiet():
    assert aa.needs_enhancement({"snr_db": 30.0, "mean_dbfs": -40.0}) is True


def test_needs_enhancement_clean_loud_false():
    # clean + loud -> raw path (no enhancement), preserving clean accuracy
    assert aa.needs_enhancement({"snr_db": 30.0, "mean_dbfs": -18.0}) is False


def test_needs_enhancement_none_profile_false():
    assert aa.needs_enhancement({"snr_db": None, "mean_dbfs": None}) is False


def teardown_module():
    importlib.reload(config)
