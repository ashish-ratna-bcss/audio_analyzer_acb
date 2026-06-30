"""L2 — lightweight, deterministic audio analysis (CPU, no model).

Estimates recording quality BEFORE any enhancement so the pipeline can decide
*whether* to enhance the ASR input, instead of guessing from loudness alone.
The key signal is SNR: the ACB sting audio is loud enough (~-23 dBFS) yet noisy
(low SNR), which a volume-only gate misses. Pure numpy over the 16 kHz track.
"""
import logging
import math

import config

logger = logging.getLogger(__name__)

_EPS = 1e-9


def _frames_rms(samples, frame=400, hop=160):
    """RMS per ~25ms frame (16kHz)."""
    import numpy as np
    n = len(samples)
    if n < frame:
        return np.array([float(np.sqrt(np.mean(samples ** 2) + _EPS))])
    idx = range(0, n - frame + 1, hop)
    return np.array([float(np.sqrt(np.mean(samples[i:i + frame] ** 2) + _EPS)) for i in idx])


def analyze(wav_path: str) -> dict:
    """Return {mean_dbfs, snr_db, clipping_ratio, reverb_proxy}. Never raises —
    on failure returns all-None so the caller falls back to volume-only gating."""
    empty = {"mean_dbfs": None, "snr_db": None, "clipping_ratio": None,
             "reverb_proxy": None}
    try:
        import numpy as np
        import soundfile as sf
        samples, sr = sf.read(wav_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if samples.size == 0:
            return empty

        rms_all = float(np.sqrt(np.mean(samples ** 2) + _EPS))
        mean_dbfs = round(20.0 * math.log10(rms_all + _EPS), 2)

        rms = _frames_rms(samples)
        # Noise floor vs speech level from the RMS distribution -> crude SNR.
        noise = float(np.percentile(rms, 10))
        speech = float(np.percentile(rms, 90))
        snr_db = round(max(0.0, min(60.0, 20.0 * math.log10((speech + _EPS) / (noise + _EPS)))), 2)

        clipping_ratio = round(float(np.mean(np.abs(samples) > 0.98)), 5)

        # Reverb proxy: clear speech has a strongly modulated energy envelope;
        # reverberation/noise smears it -> low modulation. 0=clear, 1=smeared.
        log_rms = np.log(rms + _EPS)
        modulation = float(np.std(log_rms))
        reverb_proxy = round(max(0.0, min(1.0, 1.0 - modulation / 2.0)), 3)

        return {"mean_dbfs": mean_dbfs, "snr_db": snr_db,
                "clipping_ratio": clipping_ratio, "reverb_proxy": reverb_proxy}
    except Exception as exc:
        logger.warning("audio analysis failed for %s: %s", wav_path, exc)
        return empty


def needs_enhancement(profile: dict) -> bool:
    """Decide if the ASR input should be enhanced from the analysis profile:
    low SNR (noisy) OR low loudness (quiet). Either condition marks a recording
    below the raw-decode quality bar. Clean, loud recordings return False."""
    snr = profile.get("snr_db")
    dbfs = profile.get("mean_dbfs")
    if snr is not None and snr < config.ASR_ANALYSIS_SNR_DB:
        return True
    if dbfs is not None and dbfs < config.ASR_LOW_VOLUME_DBFS:
        return True
    return False
