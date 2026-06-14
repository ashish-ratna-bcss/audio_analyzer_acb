import io
import wave
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def make_wav_bytes(duration_sec=1, sample_rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"\x00\x00" * sample_rate * duration_sec)
    return buf.getvalue()


@pytest.fixture
def client():
    with patch("services.whisper_service.load_model"), \
         patch("services.diarization_service.load_pipeline"):
        from app import app
        with TestClient(app) as test_client:
            yield test_client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_transcribe_success(client):
    mock_whisper = {
        "language": "te",
        "duration": 1.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
    }
    mock_speaker = [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}]

    wav_bytes = make_wav_bytes()

    mock_aligned = [{"speaker": "Speaker_1", "start": 0.0, "end": 1.0, "text": "hello"}]
    with patch("api.routes.stt.convert_to_wav"), \
         patch("api.routes.stt.transcribe", return_value=mock_whisper), \
         patch("api.routes.stt.diarize", return_value=mock_speaker), \
         patch("api.routes.stt.align_segments", return_value=mock_aligned):
        resp = client.post(
            "/stt/transcribe",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"language": "te", "diarize": "true"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "te"
    assert body["duration"] == 1.0
    assert len(body["segments"]) == 1
    assert body["segments"][0]["speaker"] == "Speaker_1"


def test_transcribe_unsupported_format(client):
    resp = client.post(
        "/stt/transcribe",
        files={"audio": ("test.xyz", b"fake", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_transcribe_with_translation(client):
    mock_whisper = {
        "language": "te",
        "duration": 2.0,
        "segments": [{"start": 0.0, "end": 2.0, "text": "నమస్కారం"}],
    }
    mock_speaker = [{"start": 0.0, "end": 2.0, "speaker": "Speaker_1"}]
    mock_translated = [
        {"start": 0.0, "end": 2.0, "text": "నమస్కారం",
         "speaker": "Speaker_1", "translated_text": "Hello"}
    ]

    with patch("api.routes.stt.convert_to_wav"), \
         patch("api.routes.stt.transcribe", return_value=mock_whisper), \
         patch("api.routes.stt.diarize", return_value=mock_speaker), \
         patch("api.routes.stt.translate_segments", return_value=mock_translated):
        resp = client.post(
            "/stt/transcribe",
            files={"audio": ("test.wav", make_wav_bytes(), "audio/wav")},
            data={"translate": "true"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["segments"][0]["translated_text"] == "Hello"
