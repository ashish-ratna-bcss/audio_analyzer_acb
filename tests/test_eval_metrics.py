"""Unit tests for the ASR evaluation metrics + normalization."""
from eval import metrics
from eval.normalize import normalize_text, tokens


# --- normalization ---
def test_normalize_nfc_lower_punct_whitespace():
    assert normalize_text("  Hello, WORLD!! ") == "hello world"
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_normalize_idempotent():
    once = normalize_text("Section 302, sir.")
    assert normalize_text(once) == once


def test_tokens():
    assert tokens("Recovery notice issued") == ["recovery", "notice", "issued"]


# --- WER / CER ---
def test_wer_cer_identical_zero():
    assert metrics.wer("ధర పదిహేను", "ధర పదిహేను") == 0.0
    assert metrics.cer("ధర పదిహేను", "ధర పదిహేను") == 0.0


def test_wer_full_miss_empty_hyp():
    assert metrics.wer("ఏదో మాట ఉంది", "") == 1.0


def test_wer_partial():
    w = metrics.wer("a b c d", "a b x d")  # 1 of 4 wrong
    assert 0.2 < w < 0.3


# --- Script Fidelity Rate ---
def test_sfr_pure_telugu_high():
    assert metrics.script_fidelity_rate("అందుబాటులో ఉన్నారు") == 1.0


def test_sfr_code_mix_allowed():
    # Telugu + Latin + digits all allowed
    assert metrics.script_fidelity_rate("statement 1100 ఇచ్చినా") == 1.0


def test_sfr_script_collapse_low():
    # Devanagari (Hindi script) in a Telugu transcript = collapse
    r = metrics.script_fidelity_rate("हप ह चपंड")
    assert r < 0.1


# --- number accuracy ---
def test_number_accuracy_preserved():
    assert metrics.number_accuracy("amount 1100 rupees", "1100 rupees due") == 1.0


def test_number_accuracy_wrong_digit_penalized():
    # 302 -> 307 : the reference number is missing
    assert metrics.number_accuracy("section 302", "section 307") == 0.0


def test_number_accuracy_no_numbers_is_one():
    assert metrics.number_accuracy("ఏదో మాట", "వేరే మాట") == 1.0


# --- entity accuracy ---
def test_entity_accuracy_preserved():
    g = {"CIBIL": [], "mPokket": []}
    assert metrics.entity_accuracy("your CIBIL score", "CIBIL is fine", g) == 1.0


def test_entity_accuracy_missing_entity():
    g = {"CIBIL": [], "mPokket": []}
    # ref has CIBIL, hyp lost it
    assert metrics.entity_accuracy("your CIBIL score", "score is fine", g) == 0.0


def test_entity_accuracy_no_entities_is_one():
    g = {"CIBIL": []}
    assert metrics.entity_accuracy("ఏదో మాట", "వేరే మాట", g) == 1.0


# --- score_pair shape ---
def test_score_pair_keys():
    sc = metrics.score_pair("ధర 100", "ధర 100", glossary={})
    assert set(sc) == {"wer", "cer", "sfr", "number_acc", "entity_acc"}
    assert sc["wer"] == 0.0 and sc["number_acc"] == 1.0
