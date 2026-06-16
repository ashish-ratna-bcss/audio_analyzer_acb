import os
import subprocess
import config


class UnsupportedFormatError(ValueError):
    pass


def convert_to_wav(input_path: str, output_path: str) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(f"Unsupported format: {ext}")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        # Force the first audio stream. Multi-track files (separate
        # caller/agent tracks, or a silent placeholder track) otherwise let
        # ffmpeg auto-pick the wrong/empty stream -> silence -> bad STT.
        "-map", "0:a:0",
        "-ac", str(config.TARGET_CHANNELS),
        "-ar", str(config.TARGET_SAMPLE_RATE),
        "-sample_fmt", "s16",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    return output_path


def measure_mean_volume(path: str):
    """Mean loudness of the audio in dBFS via ffmpeg volumedetect.

    Returns a float (e.g. -27.3) or None if it cannot be parsed (caller then
    defaults to VAD on).
    """
    cmd = ["ffmpeg", "-i", path, "-af", "volumedetect", "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stderr.splitlines():
        if "mean_volume:" in line:
            try:
                return float(line.split("mean_volume:")[1].split("dB")[0].strip())
            except (IndexError, ValueError):
                return None
    return None
