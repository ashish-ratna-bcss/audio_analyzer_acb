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
        "-ac", str(config.TARGET_CHANNELS),
        "-ar", str(config.TARGET_SAMPLE_RATE),
        "-sample_fmt", "s16",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    return output_path
