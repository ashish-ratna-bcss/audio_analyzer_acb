"""Unit tests for the deterministic glossary correction layer."""
import importlib

import config
from services import glossary_service


def _reload_with(glossary, enabled=True):
    config.GLOSSARY = glossary
    config.GLOSSARY_CORRECTION_ENABLED = enabled
    glossary_service._COMPILED = None  # reset compiled cache
    return glossary_service


def test_corrects_known_telugu_mishear():
    g = _reload_with({"mPokket": ["ఇంటి పక్కనుంచి", "నిపోకెటి"]})
    r = g.correct("ఇంటి పక్కనుంచి మాట్లాడుతున్నాను సర్")
    assert "mPokket" in r["text"]
    assert "ఇంటి పక్కనుంచి" not in r["text"]
    assert r["replacements"] == [{"from": "ఇంటి పక్కనుంచి", "to": "mPokket"}]


def test_ascii_alias_case_insensitive_word_boundary():
    g = _reload_with({"CIBIL": ["cibil", "Sibulhamper"]})
    r = g.correct("your Sibulhamper score and CIBIL is fine")
    # Sibulhamper -> CIBIL; existing CIBIL already canonical (no spurious change)
    assert r["text"] == "your CIBIL score and CIBIL is fine"
    assert {"from": "Sibulhamper", "to": "CIBIL"} in r["replacements"]


def test_word_boundary_no_substring_clobber():
    g = _reload_with({"EMI": ["emi"]})
    # must not corrupt 'emirates' / 'semi'
    r = g.correct("semi emirates emi")
    assert r["text"] == "semi emirates EMI"


def test_longest_alias_first():
    g = _reload_with({"recovery notice": ["recovery notis", "recovery"]})
    r = g.correct("recovery notis issued")
    assert r["text"] == "recovery notice issued"


def test_no_match_returns_original():
    g = _reload_with({"mPokket": ["ఇంటి పక్క"]})
    r = g.correct("ఏదో వేరే మాట")
    assert r["text"] == "ఏదో వేరే మాట"
    assert r["replacements"] == []


def test_disabled_is_passthrough():
    g = _reload_with({"CIBIL": ["Sibulhamper"]}, enabled=False)
    r = g.correct("Sibulhamper")
    assert r["text"] == "Sibulhamper"
    assert r["replacements"] == []


def test_empty_text_safe():
    g = _reload_with({"CIBIL": ["Sibulhamper"]})
    assert g.correct("")["text"] == ""
    assert g.correct(None)["text"] == ""


def teardown_module():
    importlib.reload(config)
    glossary_service._COMPILED = None
