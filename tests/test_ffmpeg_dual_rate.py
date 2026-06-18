import subprocess, shutil, wave
import pytest
from services.ffmpeg_service import convert_dual_rate

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make_tone(path):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1", "-ar", "22050", str(path)],
                   capture_output=True, check=True)


def test_dual_rate_outputs(tmp_path):
    src = tmp_path / "tone.wav"; _make_tone(src)
    o48 = tmp_path / "48k.wav"; o16 = tmp_path / "16k.wav"
    convert_dual_rate(str(src), str(o48), str(o16))
    with wave.open(str(o48)) as w:
        assert w.getframerate() == 48000 and w.getnchannels() == 1
    with wave.open(str(o16)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
