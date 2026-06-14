import os
import pytest
import wave
from services.ffmpeg_service import convert_to_wav, UnsupportedFormatError


def test_unsupported_format_raises():
    with pytest.raises(UnsupportedFormatError):
        convert_to_wav("/tmp/fake.xyz", "/tmp/out.wav")


def test_wav_file_conversion(tmp_path):
    # create minimal valid WAV at wrong sample rate using wave module
    src = str(tmp_path / "test_stereo_44k.wav")
    dst = str(tmp_path / "out.wav")
    with wave.open(src, "w") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(b"\x00\x00" * 44100 * 2)  # 1 second stereo
    convert_to_wav(src, dst)
    assert os.path.exists(dst)
    with wave.open(dst) as f:
        assert f.getnchannels() == 1
        assert f.getframerate() == 16000


def test_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        convert_to_wav("/tmp/does_not_exist.wav", "/tmp/out.wav")
