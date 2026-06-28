"""ASR scoring metrics — pure functions over (reference, hypothesis) strings.

All operate on normalized text (eval.normalize). Designed for Indic forensic ASR:
beyond WER/CER, they capture script collapse (SFR) and forensic-critical number /
entity preservation that plain WER understates.
"""
import re

from eval.normalize import normalize_text

# Unicode blocks that are legitimate in Telugu+English code-mixed forensic output.
# Anything outside these (Devanagari, Kannada, Tamil, Arabic…) = script collapse.
_TELUGU = (0x0C00, 0x0C7F)
_LATIN_BLOCKS = [(0x0041, 0x005A), (0x0061, 0x007A)]  # A-Z a-z
_DIGITS = (0x0030, 0x0039)


def _wer_cer(ref: str, hyp: str):
    import jiwer
    r, h = normalize_text(ref), normalize_text(hyp)
    if not r:
        return (0.0, 0.0) if not h else (1.0, 1.0)
    if not h:
        return 1.0, 1.0
    return float(jiwer.wer(r, h)), float(jiwer.cer(r, h))


def wer(ref: str, hyp: str) -> float:
    return _wer_cer(ref, hyp)[0]


def cer(ref: str, hyp: str) -> float:
    return _wer_cer(ref, hyp)[1]


def _in(cp, lo, hi):
    return lo <= cp <= hi


def script_fidelity_rate(hyp: str) -> float:
    """Fraction of *script-bearing* characters in an allowed block (Telugu,
    Latin, digits). Whitespace/punctuation ignored. Low SFR == script collapse
    (e.g. Telugu audio transcribed into Devanagari/Kannada). 1.0 when no
    script-bearing chars (nothing to violate)."""
    allowed = considered = 0
    for ch in normalize_text(hyp, strip_punct=True):
        if ch.isspace():
            continue
        cp = ord(ch)
        considered += 1
        ok = (_in(cp, *_TELUGU) or _in(cp, *_DIGITS)
              or any(_in(cp, lo, hi) for lo, hi in _LATIN_BLOCKS))
        if ok:
            allowed += 1
    return round(allowed / considered, 4) if considered else 1.0


_DIGIT_RUN = re.compile(r"\d+")


def number_accuracy(ref: str, hyp: str) -> float:
    """Recall of digit runs: forensic numbers (302, account/phone, amounts) must
    survive. Multiset match. 1.0 when the reference has no digit runs."""
    from collections import Counter
    rnums = Counter(_DIGIT_RUN.findall(ref or ""))
    if not rnums:
        return 1.0
    hnums = Counter(_DIGIT_RUN.findall(hyp or ""))
    matched = sum(min(c, hnums.get(n, 0)) for n, c in rnums.items())
    return round(matched / sum(rnums.values()), 4)


def entity_accuracy(ref: str, hyp: str, glossary) -> float:
    """Of the glossary entities present in the reference, fraction also present
    in the hypothesis (case-insensitive substring on normalized text). 1.0 when
    no glossary entity appears in the reference."""
    canon = list((glossary or {}).keys()) if isinstance(glossary, dict) else list(glossary or [])
    rn, hn = normalize_text(ref), normalize_text(hyp)
    present = [c for c in canon if normalize_text(c) and normalize_text(c) in rn]
    if not present:
        return 1.0
    hit = sum(1 for c in present if normalize_text(c) in hn)
    return round(hit / len(present), 4)


def score_pair(ref: str, hyp: str, *, glossary=None) -> dict:
    """All metrics for one (ref, hyp) pair."""
    w, c = _wer_cer(ref, hyp)
    return {
        "wer": round(w, 4), "cer": round(c, 4),
        "sfr": script_fidelity_rate(hyp),
        "number_acc": number_accuracy(ref, hyp),
        "entity_acc": entity_accuracy(ref, hyp, glossary or {}),
    }
