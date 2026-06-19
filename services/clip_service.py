import subprocess


def cut(wav_path: str, start: float, end: float, out_path: str,
        normalize: bool = False) -> str:
    dur = max(0.0, end - start)
    af = []
    if normalize:
        # dynaudnorm: per-frame dynamic loudness normalization. Lifts quiet
        # speech toward a consistent level without hard clipping loud parts,
        # so low-volume conversation decodes instead of reading as silence.
        af = ["-af", "dynaudnorm=f=200:g=15:p=0.9:m=10"]
    cmd = ["ffmpeg", "-y", "-i", wav_path, "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           *af, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"clip cut failed: {r.stderr}")
    return out_path
