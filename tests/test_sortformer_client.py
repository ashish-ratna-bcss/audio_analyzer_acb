import io
import json
from unittest.mock import patch
from services import sortformer_client as sc


def _fake_response(payload):
    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(payload).encode()
    return _Resp()


def test_client_posts_path_and_parses_segments():
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return _fake_response({"segments": [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}]})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        segs = sc.diarize_with_overlap("/data/clip.wav", num_speakers=2)

    assert segs == [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}]
    assert captured["body"] == {"audio_path": "/data/clip.wav", "num_speakers": 2}
    assert captured["url"].endswith("/diarize")


def test_client_empty_segments():
    with patch("urllib.request.urlopen", side_effect=lambda req, timeout=None: _fake_response({"segments": []})):
        assert sc.diarize_with_overlap("/data/clip.wav") == []
