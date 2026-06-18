import subprocess, shutil, wave
import pytest
from services.clip_service import cut
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")


def _tone(p):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-ar", "16000", str(p)], capture_output=True, check=True)


def test_cut_extracts_subclip(tmp_path):
    src = tmp_path / "t.wav"; _tone(src)
    out = tmp_path / "clip.wav"
    cut(str(src), 0.5, 1.5, str(out))
    with wave.open(str(out)) as w:
        dur = w.getnframes() / w.getframerate()
    assert 0.8 < dur < 1.2
