"""Unit tests for the L6b LLM enhancement service. No network: _call_llm is
monkeypatched. config flags are set per-test (conftest reloads config first)."""
import config
from services import llm_enhancement_service as llm


def _seg(text="నాకు హైదరాబాదు వెలాలి", **kw):
    base = {"text": text, "language": "te", "overlap": False, "confidence": 0.6}
    base.update(kw)
    return base


def _enable(monkeypatch, **overrides):
    monkeypatch.setattr(config, "LLM_ENHANCEMENT_ENABLED", True)
    monkeypatch.setattr(config, "LLM_ENHANCEMENT_MIN_CONF", 0.0)
    monkeypatch.setattr(config, "LLM_CORRECTION_MIN_CONF", 0.85)
    monkeypatch.setattr(config, "LLM_MAX_WORD_DELTA_RATIO", 0.15)
    for k, v in overrides.items():
        monkeypatch.setattr(config, k, v)


def _stub_llm(monkeypatch, result):
    monkeypatch.setattr(llm, "_call_llm", lambda seg: result)


# --- gating / fallback statuses -------------------------------------------

def test_disabled_returns_not_run(monkeypatch):
    monkeypatch.setattr(config, "LLM_ENHANCEMENT_ENABLED", False)
    out = llm.enhance_segment(_seg())
    assert out["correction_status"] == "not_run"
    assert out["corrected_text"] == out["original_text"]


def test_empty_text_skipped(monkeypatch):
    _enable(monkeypatch)
    out = llm.enhance_segment(_seg(text="   "))
    assert out["correction_status"] == "skipped"


def test_below_min_conf_skipped(monkeypatch):
    _enable(monkeypatch, LLM_ENHANCEMENT_MIN_CONF=0.8)
    out = llm.enhance_segment(_seg(confidence=0.5))
    assert out["correction_status"] == "skipped"


def test_provider_down_returns_error_and_raw(monkeypatch):
    _enable(monkeypatch)

    def _boom(seg):
        raise ConnectionError("ollama down")

    monkeypatch.setattr(llm, "_call_llm", _boom)
    out = llm.enhance_segment(_seg(text="hello world"))
    assert out["correction_status"] == "error"
    assert out["corrected_text"] == "hello world"


# --- correction application ------------------------------------------------

def test_corrected_applied(monkeypatch):
    _enable(monkeypatch)
    orig = "నాకు హైదరాబాదు వెలాలి"
    corr = "నాకు హైదరాబాద్ వెళ్లాలి"
    _stub_llm(monkeypatch, {
        "correction_status": "corrected", "correction_confidence": 0.94,
        "original_text": orig, "corrected_text": corr,
        "changes": [{"original": "వెలాలి", "corrected": "వెళ్లాలి", "type": "asr_error"}],
    })
    out = llm.enhance_segment(_seg(text=orig))
    assert out["correction_status"] == "corrected"
    assert out["corrected_text"] == corr
    assert out["changes"]


def test_low_correction_confidence_kept_unchanged(monkeypatch):
    _enable(monkeypatch)
    orig = "నాకు హైదరాబాదు వెలాలి"
    _stub_llm(monkeypatch, {
        "correction_status": "corrected", "correction_confidence": 0.5,
        "original_text": orig, "corrected_text": "నాకు హైదరాబాద్ వెళ్లాలి",
        "changes": [],
    })
    out = llm.enhance_segment(_seg(text=orig))
    assert out["correction_status"] == "unchanged"
    assert out["corrected_text"] == orig


def test_model_says_unchanged(monkeypatch):
    _enable(monkeypatch)
    orig = "all good"
    _stub_llm(monkeypatch, {
        "correction_status": "unchanged", "correction_confidence": 1.0,
        "original_text": orig, "corrected_text": orig, "changes": [],
    })
    out = llm.enhance_segment(_seg(text=orig))
    assert out["correction_status"] == "unchanged"


# --- deterministic guards --------------------------------------------------

def test_guard_rejects_translation_script_change(monkeypatch):
    _enable(monkeypatch)
    orig = "నాకు హైదరాబాద్"          # Telugu
    _stub_llm(monkeypatch, {
        "correction_status": "corrected", "correction_confidence": 0.99,
        "original_text": orig, "corrected_text": "I need Hyderabad",  # translated to Latin
        "changes": [],
    })
    out = llm.enhance_segment(_seg(text=orig))
    assert out["correction_status"] == "guard_rejected"
    assert out["corrected_text"] == orig


def test_guard_rejects_new_indic_script(monkeypatch):
    _enable(monkeypatch)
    orig = "నాకు హైదరాబాద్"          # Telugu
    _stub_llm(monkeypatch, {
        "correction_status": "corrected", "correction_confidence": 0.99,
        "original_text": orig, "corrected_text": "मुझे हैदराबाद",  # Devanagari
        "changes": [],
    })
    out = llm.enhance_segment(_seg(text=orig))
    assert out["correction_status"] == "guard_rejected"


def test_guard_rejects_number_change(monkeypatch):
    _enable(monkeypatch)
    orig = "account 1234567890 lo"
    _stub_llm(monkeypatch, {
        "correction_status": "corrected", "correction_confidence": 0.99,
        "original_text": orig, "corrected_text": "account 9999999999 lo",
        "changes": [],
    })
    out = llm.enhance_segment(_seg(text=orig))
    assert out["correction_status"] == "guard_rejected"


def test_guard_preserves_number_passes(monkeypatch):
    _enable(monkeypatch)
    orig = "akount 1234567890 lo"
    corr = "account 1234567890 lo"   # spelling fix, number intact, same word count
    _stub_llm(monkeypatch, {
        "correction_status": "corrected", "correction_confidence": 0.99,
        "original_text": orig, "corrected_text": corr, "changes": [],
    })
    out = llm.enhance_segment(_seg(text=orig))
    assert out["correction_status"] == "corrected"
    assert out["corrected_text"] == corr


def test_guard_rejects_word_addition(monkeypatch):
    _enable(monkeypatch)
    orig = "I will come"
    _stub_llm(monkeypatch, {
        "correction_status": "corrected", "correction_confidence": 0.99,
        "original_text": orig, "corrected_text": "I will come tomorrow",
        "changes": [],
    })
    out = llm.enhance_segment(_seg(text=orig, language="en"))
    assert out["correction_status"] == "guard_rejected"


def test_guard_rejects_blanking(monkeypatch):
    _enable(monkeypatch)
    orig = "hello"
    _stub_llm(monkeypatch, {
        "correction_status": "corrected", "correction_confidence": 0.99,
        "original_text": orig, "corrected_text": "", "changes": [],
    })
    out = llm.enhance_segment(_seg(text=orig, language="en"))
    assert out["correction_status"] == "guard_rejected"


def test_overlap_rejects_word_count_change(monkeypatch):
    _enable(monkeypatch)
    orig = "ఒకటి రెండు"
    _stub_llm(monkeypatch, {
        "correction_status": "corrected", "correction_confidence": 0.99,
        "original_text": orig, "corrected_text": "ఒకటి రెండు మూడు",  # added a word
        "changes": [],
    })
    out = llm.enhance_segment(_seg(text=orig, overlap=True))
    assert out["correction_status"] == "guard_rejected"


def test_code_mixed_telugu_english_allowed(monkeypatch):
    _enable(monkeypatch)
    orig = "అన్నా acount లో transfer చేయి"
    corr = "అన్నా account లో transfer చేయి"   # English spelling fix, Telugu intact
    _stub_llm(monkeypatch, {
        "correction_status": "corrected", "correction_confidence": 0.95,
        "original_text": orig, "corrected_text": corr, "changes": [],
    })
    out = llm.enhance_segment(_seg(text=orig))
    assert out["correction_status"] == "corrected"
    assert out["corrected_text"] == corr


# --- JSON parse helper -----------------------------------------------------

def test_parse_llm_json_tolerates_wrapping_text():
    raw = 'Here is the result:\n{"correction_status":"unchanged","corrected_text":"x"}\nDone.'
    parsed = llm._parse_llm_json(raw)
    assert parsed["correction_status"] == "unchanged"


# --- primary -> fallback Ollama routing -----------------------------------

def test_ollama_uses_primary_when_ok(monkeypatch):
    monkeypatch.setattr(config, "LLM_OLLAMA_URL", "http://primary:11434")
    monkeypatch.setattr(config, "LLM_MODEL", "m14b")
    calls = []

    def _gen(url, model, seg):
        calls.append((url, model))
        return {"correction_status": "unchanged", "corrected_text": seg["text"]}

    monkeypatch.setattr(llm, "_ollama_generate", _gen)
    llm._call_ollama(_seg(text="x"))
    assert calls == [("http://primary:11434", "m14b")]


def test_ollama_falls_back_on_primary_failure(monkeypatch):
    monkeypatch.setattr(config, "LLM_OLLAMA_URL", "http://primary:11434")
    monkeypatch.setattr(config, "LLM_MODEL", "m14b")
    monkeypatch.setattr(config, "LLM_OLLAMA_FALLBACK_URL", "http://fallback:11434")
    monkeypatch.setattr(config, "LLM_FALLBACK_MODEL", "m7b")
    calls = []

    def _gen(url, model, seg):
        calls.append((url, model))
        if url == "http://primary:11434":
            raise ConnectionError("primary down")
        return {"correction_status": "unchanged", "corrected_text": seg["text"]}

    monkeypatch.setattr(llm, "_ollama_generate", _gen)
    llm._call_ollama(_seg(text="x"))
    assert calls == [("http://primary:11434", "m14b"), ("http://fallback:11434", "m7b")]


def test_ollama_both_fail_raises_to_error_status(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(config, "LLM_OLLAMA_FALLBACK_URL", "http://fallback:11434")
    monkeypatch.setattr(config, "LLM_FALLBACK_MODEL", "m7b")

    def _gen(url, model, seg):
        raise ConnectionError("down")

    monkeypatch.setattr(llm, "_ollama_generate", _gen)
    out = llm.enhance_segment(_seg(text="hello world"))
    assert out["correction_status"] == "error"
    assert out["corrected_text"] == "hello world"
