"""Uniform robust preprocessing: one clean clip per unit, fed to all 3 ASR models.

Chain: cut from the DeepFilterNet3-enhanced full file -> EBU R128 loudness
normalize -> trim leading/trailing silence (never cut interior). The raw cut
from the ORIGINAL file is always kept unprocessed for audit. Every step is
non-fatal: a failure falls back to the previous clip, never raises.
"""
import os
import subprocess

import config
from services import clip_service, vad_service


def _loudnorm(in_path: str, out_path: str) -> bool:
    """EBU R128 loudness normalization. Returns False on ffmpeg failure."""
    cmd = ["ffmpeg", "-y", "-i", in_path,
           "-af", f"loudnorm=I={config.LOUDNORM_LUFS}:TP=-1.5:LRA=11",
           "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0 and os.path.exists(out_path)
    except Exception:
        return False


def prepare_clip(enhanced_src, original_src, start, end, workdir, idx, speaker):
    """Produce {'clean': path, 'raw': path} for one transcription unit.

    'clean' is the enhanced cut, loudness-normalized and edge-trimmed, fed to all
    three ASR models. 'raw' is the untouched original cut, retained for audit.
    """
    raw = os.path.join(workdir, f"seg_{idx:04d}_{speaker}_org.wav")
    clip_service.cut(original_src, start, end, raw, normalize=False)

    enh_cut = os.path.join(workdir, f"seg_{idx:04d}_{speaker}_enh.wav")
    clip_service.cut(enhanced_src, start, end, enh_cut, normalize=False)

    # Loudness normalize (fall back to enh_cut if it fails).
    loud = os.path.join(workdir, f"seg_{idx:04d}_{speaker}_loud.wav")
    clean = loud if _loudnorm(enh_cut, loud) else enh_cut

    # Trim leading/trailing silence to detected speech span (keep interior).
    speech = vad_service.detect_speech(clean)
    if speech:
        s0 = min(seg["start"] for seg in speech)
        e0 = max(seg["end"] for seg in speech)
        if e0 - s0 > 0.1:
            trimmed = os.path.join(workdir, f"seg_{idx:04d}_{speaker}_clean.wav")
            try:
                clip_service.cut(clean, s0, e0, trimmed, normalize=False)
                clean = trimmed
            except Exception:
                pass  # keep untrimmed clean clip

    return {"clean": clean, "raw": raw}
