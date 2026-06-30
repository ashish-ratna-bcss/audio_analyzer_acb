"""Optional dereverberation for far-field / reverberant recordings (L3).

WPE (Weighted Prediction Error) — a classical, deterministic dereverb algorithm
(no neural model, no hallucination risk → forensically safe). Off by default
(ASR_DEREVERB); only runs when enabled AND nara_wpe is installed. Best-effort:
returns False on any failure so the pipeline falls back to the enhanced track.
"""
import logging

import config

logger = logging.getLogger(__name__)


def available() -> bool:
    try:
        import nara_wpe  # noqa: F401
        return True
    except Exception:
        return False


def dereverb(in_path: str, out_path: str) -> bool:
    """WPE-dereverberate a 16kHz mono WAV. Returns True on success, else False
    (caller keeps the un-dereverbed input). Never raises."""
    if not config.ASR_DEREVERB:
        return False
    try:
        import numpy as np
        import soundfile as sf
        from nara_wpe.wpe import wpe
        from nara_wpe.utils import stft, istft

        y, sr = sf.read(in_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        # WPE expects (channels, frames, freq); single channel here.
        Y = stft(y[None, :], size=512, shift=128)              # (1, T, F)
        Y = Y.transpose(2, 0, 1)                               # (F, 1, T)
        Z = wpe(Y, taps=10, delay=3, iterations=3)             # dereverbed
        z = istft(Z.transpose(1, 2, 0)[0], size=512, shift=128)
        z = np.asarray(z, dtype="float32")
        peak = float(np.max(np.abs(z)) or 1.0)
        if peak > 1.0:
            z = z / peak
        sf.write(out_path, z, sr if sr else 16000)
        return True
    except Exception as exc:
        logger.warning("dereverb failed for %s: %s", in_path, exc)
        return False
