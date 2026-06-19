from unittest.mock import patch
from services import indic_asr_service as ias


def test_abstains_on_non_indic():
    r = ias.transcribe_clip("/tmp/x.wav", "ko")
    assert r["abstained"] is True
    assert r["text"] == ""
    assert r["confidence"] is None
    assert r["model"] == "indic_unsupported"


def test_runs_for_indic_and_returns_text():
    import torch

    class _M:
        def __call__(self, wav, lang, mode):
            return ["పదిహేను"]

        def eval(self):
            return self

        def to(self, d):
            return self

    with patch.object(ias, "_load", return_value=_M()), \
         patch("torchaudio.load", return_value=(torch.zeros(1, 16000), 16000)):
        r = ias.transcribe_clip("/tmp/x.wav", "te")
    assert r["abstained"] is False
    assert r["text"] == "పదిహేను"
    assert r["confidence"] is None        # CTC checkpoint exposes no score -> unscored, not faked
    assert r["language"] == "te"
