import subprocess


def cut(wav_path: str, start: float, end: float, out_path: str) -> str:
    dur = max(0.0, end - start)
    cmd = ["ffmpeg", "-y", "-i", wav_path, "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"clip cut failed: {r.stderr}")
    return out_path
