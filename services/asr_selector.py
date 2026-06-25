"""Dual-engine ASR output selection — pure, I/O-free, fully unit-testable.

Given IndicConformer and Whisper outputs for the same clip, pick the better text
by inspecting the actual content (not a noisy LID guess):

  1. Whisper text contains Latin letters or digits  -> Whisper wins
       (code-mixed English / numbers / entities — IndicConformer transliterates
        these into garbled native script).
  2. Both are pure native script (no Latin/digit):
       LaBSE similarity >= agreement_min  -> IndicConformer wins (sharper native
                                             script fidelity on pure Telugu)
       similarity  <  agreement_min       -> keep Whisper, FLAG (engines disagree)
  3. IndicConformer abstained / empty               -> Whisper wins
  4. Whisper empty or high no_speech_prob           -> IndicConformer fallback

The embedding function is injected (embed_fn) so this module stays pure and tests
can pass a deterministic fake.
"""
import re

_LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]")


def has_latin_or_digit(text: str) -> bool:
    """True if the text contains any ASCII letter or digit — the signal for
    code-mixed English, numbers, or Latin-script entities."""
    return bool(_LATIN_OR_DIGIT.search(text or ""))


def _whisper(text, confidence, *, reason, agreement=None, flag=None):
    return {"text": text, "source": "whisper", "confidence": confidence,
            "agreement": agreement, "reason": reason, "flag": flag}


def _telugu(text, confidence, *, reason, agreement=None, flag=None):
    return {"text": text, "source": "telugu_whisper", "confidence": confidence,
            "agreement": agreement, "reason": reason, "flag": flag}


def _indic(text, source, agreement, *, reason, flag=None):
    return {"text": text, "source": source, "confidence": agreement,
            "agreement": agreement, "reason": reason, "flag": flag}


def select(*, indic_text, indic_source, indic_agreement, indic_abstained,
           whisper_text, whisper_confidence, whisper_no_speech,
           embed_fn, agreement_min=0.6, no_speech_max=0.6,
           telugu_text=None, telugu_confidence=None):
    """Return the winning ASR record across up to three engines.

    Priority:
      1. Generic Whisper carries Latin/digits  -> generic Whisper (code-mix /
         numbers / entities — its strength).
      2. Else a fine-tuned native engine (telugu_text) is available -> use it
         (best pure-native acoustic accuracy), cross-checked against
         IndicConformer for a disagreement flag.
      3. Else fall back to the two-way generic-Whisper vs IndicConformer logic.

    Keys: text, source ("whisper"|"telugu_whisper"|"indic_enhanced"|...),
    confidence, agreement (None when not embedding-compared), reason, flag.
    `telugu_text=None` reduces to the original two-way behaviour.
    """
    indic_text = (indic_text or "").strip()
    whisper_text = (whisper_text or "").strip()
    telugu_text = (telugu_text or "").strip()
    whisper_usable = bool(whisper_text) and (
        whisper_no_speech is None or whisper_no_speech <= no_speech_max)

    # 1. Generic Whisper output carries Latin/digits -> code-mix or numbers.
    if whisper_usable and has_latin_or_digit(whisper_text):
        return _whisper(whisper_text, whisper_confidence,
                        reason="code_mix_or_numbers")

    # 2. Pure-native turn with a fine-tuned engine result -> prefer it.
    if telugu_text:
        if indic_text and not indic_abstained:
            try:
                sim = round(float(embed_fn(telugu_text, indic_text)), 3)
            except Exception:
                sim = 0.0
            flag = None if sim >= agreement_min else "asr_engine_disagreement"
            return _telugu(telugu_text, telugu_confidence, agreement=sim,
                           reason="telugu_ft", flag=flag)
        return _telugu(telugu_text, telugu_confidence, reason="telugu_ft_only")

    # 3a. Generic Whisper unusable and no fine-tune -> IndicConformer fallback.
    if not whisper_usable:
        return _indic(indic_text, indic_source, indic_agreement,
                      reason="whisper_unusable")

    # 3b. IndicConformer abstained or empty -> generic Whisper.
    if indic_abstained or not indic_text:
        return _whisper(whisper_text, whisper_confidence, reason="indic_abstain")

    # 3c. Both pure native script — agreement decides.
    try:
        sim = round(float(embed_fn(indic_text, whisper_text)), 3)
    except Exception:
        sim = 0.0
    if sim >= agreement_min:
        return _indic(indic_text, indic_source, sim, reason="pure_native_agree")
    return _whisper(whisper_text, whisper_confidence, reason="pure_native_disagree",
                    agreement=sim, flag="asr_engine_disagreement")
