"""L7 — deterministic domain-glossary correction (additive, forensic-safe).

Post-ASR, presentation-layer only: it NEVER mutates the persisted raw ASR text.
The caller stores the corrected string in a separate field so a reviewer always
sees both raw and corrected. Correction is a curated, explicit alias -> canonical
map (loan / debt-recovery domain) — NO fuzzy/statistical matching, so it cannot
hallucinate: a token changes only if it exactly matches a hand-listed variant.

Matching is longest-alias-first to handle multi-word variants (e.g. the ASR
mishears of "mPokket" / "CIBIL") before single tokens. ASCII aliases match
case-insensitively on word boundaries; Indic-script aliases match as exact
substrings (Telugu has no ASCII word boundaries). Every replacement is recorded
for the audit trail.
"""
import re

import config


def _compile(glossary: dict):
    """Build (pattern, canonical, alias) tuples, longest alias first."""
    entries = []
    for canonical, aliases in (glossary or {}).items():
        for alias in aliases:
            a = alias.strip()
            if not a:
                continue
            if a.isascii():
                # word-boundary, case-insensitive for Latin/code-mix tokens
                pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])",
                                 re.IGNORECASE)
            else:
                # plain substring for Indic script (no ASCII word boundaries)
                pat = re.compile(re.escape(a))
            entries.append((len(a), pat, canonical, a))
    # longest alias first so multi-word variants win over their sub-tokens
    entries.sort(key=lambda e: e[0], reverse=True)
    return [(p, c, a) for _, p, c, a in entries]


_COMPILED = None


def _patterns():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = _compile(getattr(config, "GLOSSARY", {}))
    return _COMPILED


def correct(text: str) -> dict:
    """Return {"text": corrected, "replacements": [{"from","to"}...]}.

    `text` unchanged (and replacements empty) when the layer is disabled, the
    glossary is empty, or nothing matched. Never raises."""
    original = text or ""
    if not original.strip() or not config.GLOSSARY_CORRECTION_ENABLED:
        return {"text": original, "replacements": []}

    out = original
    replacements = []
    for pat, canonical, alias in _patterns():
        if pat.search(out):
            new = pat.sub(canonical, out)
            if new != out:
                replacements.append({"from": alias, "to": canonical})
                out = new
    return {"text": out, "replacements": replacements}
