"""Indic-aware text normalization for ASR scoring.

Applied identically to reference and hypothesis before WER/CER so the metric
reflects real recognition errors, not cosmetic differences (Unicode form,
punctuation, casing, whitespace). Pure functions, no I/O.
"""
import re
import unicodedata

# Punctuation stripped before scoring (both scripts). Telugu danda/double-danda
# included alongside ASCII punctuation. Digits and letters are preserved.
_PUNCT = re.compile(
    r"[\.,!?;:\"'`~@#\$%\^&\*\(\)\[\]\{\}<>/\\\|_\-\+=…।॥।॥]")
_WS = re.compile(r"\s+")


def normalize_text(text: str, *, strip_punct: bool = True) -> str:
    """Canonicalize for comparison:
    - Unicode NFC (compose Telugu matras consistently)
    - lowercase ASCII (Telugu has no case; English code-mix normalized)
    - strip punctuation (optional) and collapse whitespace
    Returns "" for falsy input."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.lower()
    if strip_punct:
        t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def tokens(text: str) -> list:
    """Whitespace word tokens of normalized text (for WER-style word ops)."""
    n = normalize_text(text)
    return n.split() if n else []
