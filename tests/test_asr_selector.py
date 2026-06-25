"""Unit tests for the dual-engine ASR output selector — pure, no models."""
from services import asr_selector


def _embed(score):
    return lambda a, b: score


def test_has_latin_or_digit():
    assert asr_selector.has_latin_or_digit("statement ఇచ్చినా")
    assert asr_selector.has_latin_or_digit("1100 రూపాయలు")
    assert not asr_selector.has_latin_or_digit("అందుబాటులో ఉన్నారు")
    assert not asr_selector.has_latin_or_digit("")


def test_code_mix_picks_whisper():
    """Whisper output with English/Latin -> Whisper wins (code-mix/entities)."""
    r = asr_selector.select(
        indic_text="స్టేట్మెంట్ ఇచ్చినా", indic_source="indic_enhanced",
        indic_agreement=0.9, indic_abstained=False,
        whisper_text="statement ఇచ్చినా bank లో", whisper_confidence=0.8,
        whisper_no_speech=0.1, embed_fn=_embed(0.9))
    assert r["source"] == "whisper"
    assert r["reason"] == "code_mix_or_numbers"
    assert r["text"] == "statement ఇచ్చినా bank లో"
    assert r["flag"] is None


def test_numbers_pick_whisper():
    r = asr_selector.select(
        indic_text="పదకొండు వందల", indic_source="indic_enhanced",
        indic_agreement=0.9, indic_abstained=False,
        whisper_text="1100 రూపాయలు", whisper_confidence=0.85,
        whisper_no_speech=0.0, embed_fn=_embed(0.9))
    assert r["source"] == "whisper"
    assert r["reason"] == "code_mix_or_numbers"


def test_pure_telugu_agree_picks_indic():
    """Both pure native script and they agree -> IndicConformer (sharper script)."""
    r = asr_selector.select(
        indic_text="అందుబాటులో ఉన్నారు", indic_source="indic_enhanced",
        indic_agreement=0.9, indic_abstained=False,
        whisper_text="అందుబాటులో ఉన్నారా", whisper_confidence=0.7,
        whisper_no_speech=0.1, embed_fn=_embed(0.85), agreement_min=0.6)
    assert r["source"] == "indic_enhanced"
    assert r["reason"] == "pure_native_agree"
    assert r["agreement"] == 0.85
    assert r["flag"] is None


def test_pure_telugu_disagree_keeps_whisper_and_flags():
    r = asr_selector.select(
        indic_text="అందుబాటులో ఉన్నారు", indic_source="indic_enhanced",
        indic_agreement=0.9, indic_abstained=False,
        whisper_text="వేరే మాట పూర్తిగా", whisper_confidence=0.7,
        whisper_no_speech=0.1, embed_fn=_embed(0.2), agreement_min=0.6)
    assert r["source"] == "whisper"
    assert r["reason"] == "pure_native_disagree"
    assert r["flag"] == "asr_engine_disagreement"
    assert r["agreement"] == 0.2


def test_indic_abstain_picks_whisper():
    r = asr_selector.select(
        indic_text="", indic_source="none",
        indic_agreement=0.0, indic_abstained=True,
        whisper_text="ఏదో మాట", whisper_confidence=0.6,
        whisper_no_speech=0.1, embed_fn=_embed(0.0))
    assert r["source"] == "whisper"
    assert r["reason"] == "indic_abstain"


def test_whisper_empty_falls_back_to_indic():
    r = asr_selector.select(
        indic_text="అందుబాటులో", indic_source="indic_enhanced",
        indic_agreement=0.8, indic_abstained=False,
        whisper_text="", whisper_confidence=None,
        whisper_no_speech=None, embed_fn=_embed(0.9))
    assert r["source"] == "indic_enhanced"
    assert r["reason"] == "whisper_unusable"
    assert r["text"] == "అందుబాటులో"


def test_whisper_high_no_speech_falls_back_to_indic():
    r = asr_selector.select(
        indic_text="అందుబాటులో", indic_source="indic_original",
        indic_agreement=0.8, indic_abstained=False,
        whisper_text="garbage", whisper_confidence=0.3,
        whisper_no_speech=0.95, embed_fn=_embed(0.9), no_speech_max=0.6)
    assert r["source"] == "indic_original"
    assert r["reason"] == "whisper_unusable"


def test_telugu_ft_wins_pure_native():
    """Pure-native turn with a fine-tuned engine result -> Telugu-FT wins,
    cross-checked against IndicConformer (agree -> no flag)."""
    r = asr_selector.select(
        indic_text="అందుబాటులో ఉన్నారు", indic_source="indic_enhanced",
        indic_agreement=0.9, indic_abstained=False,
        whisper_text="అందుబాటులో ఉన్నారా", whisper_confidence=0.6,
        whisper_no_speech=0.1, telugu_text="అందుబాటులో ఉన్నారు", telugu_confidence=None,
        embed_fn=_embed(0.9), agreement_min=0.6)
    assert r["source"] == "telugu_whisper"
    assert r["reason"] == "telugu_ft"
    assert r["text"] == "అందుబాటులో ఉన్నారు"
    assert r["flag"] is None


def test_telugu_ft_disagree_with_indic_flags():
    r = asr_selector.select(
        indic_text="వేరే మాట పూర్తిగా", indic_source="indic_enhanced",
        indic_agreement=0.9, indic_abstained=False,
        whisper_text="ఇంకేదో మాట", whisper_confidence=0.6,
        whisper_no_speech=0.1, telugu_text="అందుబాటులో ఉన్నారు", telugu_confidence=None,
        embed_fn=_embed(0.2), agreement_min=0.6)
    assert r["source"] == "telugu_whisper"
    assert r["flag"] == "asr_engine_disagreement"


def test_code_mix_beats_telugu_ft():
    """Even with a Telugu-FT result, Latin/digits in generic Whisper win."""
    r = asr_selector.select(
        indic_text="పదకొండు వందల", indic_source="indic_enhanced",
        indic_agreement=0.9, indic_abstained=False,
        whisper_text="1100 rupees", whisper_confidence=0.8,
        whisper_no_speech=0.0, telugu_text="పదకొండు వందల", telugu_confidence=None,
        embed_fn=_embed(0.9))
    assert r["source"] == "whisper"
    assert r["reason"] == "code_mix_or_numbers"


def test_telugu_ft_only_when_indic_abstains():
    r = asr_selector.select(
        indic_text="", indic_source="none",
        indic_agreement=0.0, indic_abstained=True,
        whisper_text="", whisper_confidence=None,
        whisper_no_speech=None, telugu_text="అందుబాటులో ఉన్నారు", telugu_confidence=None,
        embed_fn=_embed(0.0))
    assert r["source"] == "telugu_whisper"
    assert r["reason"] == "telugu_ft_only"


def test_embed_failure_treated_as_disagreement():
    def _boom(a, b):
        raise RuntimeError("embed down")
    r = asr_selector.select(
        indic_text="అందుబాటులో ఉన్నారు", indic_source="indic_enhanced",
        indic_agreement=0.9, indic_abstained=False,
        whisper_text="వేరే మాట", whisper_confidence=0.7,
        whisper_no_speech=0.1, embed_fn=_boom, agreement_min=0.6)
    assert r["source"] == "whisper"
    assert r["flag"] == "asr_engine_disagreement"
