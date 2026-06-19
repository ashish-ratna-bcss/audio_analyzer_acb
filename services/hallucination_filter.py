"""Blank ASR passes that are non-speech hallucinations rather than real content.

Pure functions (no model/IO) so they are fully unit-testable. Applied to every
ASR pass output before cross-model comparison.
"""
import re

import config


def has_repetition(text: str) -> bool:
    """Detect degenerate hallucination loops: consecutive repeats or extreme monotony."""
    if not text:
        return False
    words = text.split()
    if len(words) < 4:
        return False
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return True
    if len(words) >= 8 and len(set(words)) / len(words) < 0.30:
        return True
    return False


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\[\]()]", "", text.lower())).strip()


_GHOSTS = {_normalize(p) for p in config.GHOST_PHRASES}


def filter_pass(result: dict, *, no_speech_prob: float | None = None) -> dict:
    """Return result unchanged, or blanked with a `hallucination` reason."""
    text = (result.get("text") or "").strip()

    if no_speech_prob is not None and no_speech_prob > config.NO_SPEECH_MAX:
        return {**result, "text": "", "confidence": 0.0, "hallucination": "no_speech"}

    if text and _normalize(text) in _GHOSTS:
        return {**result, "text": "", "confidence": 0.0, "hallucination": "ghost_phrase"}

    if has_repetition(text):
        return {**result, "text": "", "confidence": 0.0, "hallucination": "repetition"}

    return result
