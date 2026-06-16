import io
import wave
import pytest
from unittest.mock import patch
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


# Whisper is patched to return native text for the transcribe pass and English
# for the translate pass, so raw and english blocks are distinguishable.
def _fake_transcribe(wav_path, language="auto", use_vad=True, task="transcribe"):
    text = "hello" if task == "translate" else "హలో"
    return {"language": "te", "duration": 1.0,
            "segments": [{"start": 0.0, "end": 1.0, "text": text,
                          "confidence": 0.9, "no_speech_prob": 0.1,
                          "compression_ratio": 1.5}]}


def _fake_align(whisper_segments, speaker_segs):
    return [{**s, "speaker": "Speaker_1"} for s in whisper_segments]


def test_transcribe_returns_raw_and_english(client):
    mock_speaker = [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}]
    with patch("api.routes.stt.convert_to_wav"), \
         patch("api.routes.stt.measure_mean_volume", return_value=-20.0), \
         patch("api.routes.stt.transcribe", side_effect=_fake_transcribe), \
         patch("api.routes.stt.diarize", return_value=mock_speaker), \
         patch("api.routes.stt.align_segments", side_effect=_fake_align):
        resp = client.post(
            "/stt/transcribe",
            files={"audio": ("test.wav", make_wav_bytes(), "audio/wav")},
            data={"language": "te"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "te"
    assert body["duration"] == 1.0
    # both diarized blocks present
    assert body["raw"]["dialogue"][0]["speaker"] == "Speaker_1"
    assert body["raw"]["dialogue"][0]["text"] == "హలో"
    assert body["english"]["dialogue"][0]["text"] == "hello"
    # default mode: no debug metrics
    assert "confidence" not in body["raw"]["dialogue"][0]
    assert "segments" not in body["raw"]


def test_transcribe_debug_adds_metrics(client):
    mock_speaker = [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}]
    with patch("api.routes.stt.convert_to_wav"), \
         patch("api.routes.stt.measure_mean_volume", return_value=-20.0), \
         patch("api.routes.stt.transcribe", side_effect=_fake_transcribe), \
         patch("api.routes.stt.diarize", return_value=mock_speaker), \
         patch("api.routes.stt.align_segments", side_effect=_fake_align):
        resp = client.post(
            "/stt/transcribe",
            files={"audio": ("test.wav", make_wav_bytes(), "audio/wav")},
            data={"language": "te", "debug": "true"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["raw"]["dialogue"][0]["confidence"] == 0.9
    assert body["raw"]["segments"][0]["compression_ratio"] == 1.5
    assert body["english"]["segments"][0]["text"] == "hello"


def test_transcribe_diarize_false_single_speaker(client):
    # diarize=false: pyannote skipped, everything under Speaker_1.
    with patch("api.routes.stt.convert_to_wav"), \
         patch("api.routes.stt.measure_mean_volume", return_value=-20.0), \
         patch("api.routes.stt.transcribe", side_effect=_fake_transcribe), \
         patch("api.routes.stt.diarize") as mock_diarize, \
         patch("api.routes.stt.align_segments", side_effect=_fake_align):
        resp = client.post(
            "/stt/transcribe",
            files={"audio": ("test.wav", make_wav_bytes(), "audio/wav")},
            data={"language": "te", "diarize": "false"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["raw"]["dialogue"][0]["speaker"] == "Speaker_1"
    mock_diarize.assert_not_called()  # diarization skipped


def test_transcribe_unsupported_format(client):
    resp = client.post(
        "/stt/transcribe",
        files={"audio": ("test.xyz", b"fake", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_transcribe_requires_api_key_when_set(client, monkeypatch):
    monkeypatch.setattr("config.API_KEY", "secret")
    # missing key -> 401, before any processing
    resp = client.post(
        "/stt/transcribe",
        files={"audio": ("test.wav", make_wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 401
    # wrong key -> 401
    resp = client.post(
        "/stt/transcribe",
        files={"audio": ("test.wav", make_wav_bytes(), "audio/wav")},
        headers={"X-API-Key": "nope"},
    )
    assert resp.status_code == 401


def test_transcribe_accepts_valid_api_key(client, monkeypatch):
    monkeypatch.setattr("config.API_KEY", "secret")
    mock_speaker = [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}]
    with patch("api.routes.stt.convert_to_wav"), \
         patch("api.routes.stt.measure_mean_volume", return_value=-20.0), \
         patch("api.routes.stt.transcribe", side_effect=_fake_transcribe), \
         patch("api.routes.stt.diarize", return_value=mock_speaker), \
         patch("api.routes.stt.align_segments", side_effect=_fake_align):
        resp = client.post(
            "/stt/transcribe",
            files={"audio": ("test.wav", make_wav_bytes(), "audio/wav")},
            headers={"X-API-Key": "secret"},
        )
    assert resp.status_code == 200
