from unittest.mock import patch
from services import preprocess_service as pp


def test_prepare_clip_pipeline_wires_steps(tmp_path):
    calls = {"cut": 0, "loud": 0, "vad": 0}

    def fake_cut(src, s, e, out, normalize=False):
        calls["cut"] += 1
        open(out, "wb").close()
        return out

    def fake_loud(in_p, out_p):
        calls["loud"] += 1
        open(out_p, "wb").close()
        return True

    def fake_vad(path):
        calls["vad"] += 1
        return [{"start": 0.2, "end": 1.8}]   # trim edges to detected speech

    with patch.object(pp.clip_service, "cut", side_effect=fake_cut), \
         patch.object(pp, "_loudnorm", side_effect=fake_loud), \
         patch.object(pp.vad_service, "detect_speech", side_effect=fake_vad):
        out = pp.prepare_clip(str(tmp_path / "enh.wav"), str(tmp_path / "org.wav"),
                              10.0, 13.0, str(tmp_path), 0, "Speaker_1")

    assert "clean" in out and "raw" in out
    assert calls["loud"] == 1
    assert calls["vad"] >= 1
    assert calls["cut"] >= 2   # at least raw + enhanced cuts


def test_loudnorm_failure_keeps_clip(tmp_path):
    def fake_cut(src, s, e, out, normalize=False):
        open(out, "wb").close()
        return out

    with patch.object(pp.clip_service, "cut", side_effect=fake_cut), \
         patch.object(pp, "_loudnorm", return_value=False), \
         patch.object(pp.vad_service, "detect_speech", return_value=[]):
        out = pp.prepare_clip(str(tmp_path / "enh.wav"), str(tmp_path / "org.wav"),
                              0.0, 2.0, str(tmp_path), 1, "Speaker_2")
    # No crash; clean path still returned (falls back to pre-loudnorm cut).
    assert out["clean"]
    assert out["raw"]
