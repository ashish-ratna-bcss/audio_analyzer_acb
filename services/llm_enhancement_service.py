"""L6b — Multilingual LLM transcript enhancement.

Corrects ONLY obvious ASR character/spelling mistakes on an existing transcript
segment, preserving meaning, language, script, numbers, speakers, timestamps and
boundaries. Purely additive: the caller stores the returned record under
candidates["llm_enhancement"] and NEVER overwrites seg.text.

`enhance_segment` always returns a correction record and never raises — if the
provider is down/disabled/uncertain or a deterministic guard rejects the output,
the original text is returned with a `correction_status` explaining why. This
keeps the pipeline behaving exactly as it did before this layer existed whenever
the LLM is unavailable.

Heavy/network calls are isolated: Ollama via stdlib urllib (no laptop dep),
Anthropic via lazy import. On the build box this module is stubbed in
tests/conftest.py.
"""
import json
import re
import logging

import config

logger = logging.getLogger(__name__)

# Unicode block ranges for the scripts of the supported Indian languages. Used by
# the script-preservation guard to forbid translation / script conversion. Latin
# and digits are intentionally excluded so code-mixing (Indic + English/numbers)
# is preserved, not penalised.
_INDIC_BLOCKS = {
    "devanagari": (0x0900, 0x097F),   # Hindi, Marathi, Sanskrit
    "bengali":    (0x0980, 0x09FF),   # Bengali, Assamese
    "gurmukhi":   (0x0A00, 0x0A7F),   # Punjabi
    "gujarati":   (0x0A80, 0x0AFF),
    "oriya":      (0x0B00, 0x0B7F),   # Odia
    "tamil":      (0x0B80, 0x0BFF),
    "telugu":     (0x0C00, 0x0C7F),
    "kannada":    (0x0C80, 0x0CFF),
    "malayalam":  (0x0D00, 0x0D7F),
    "arabic":     (0x0600, 0x06FF),   # Urdu
}

_DIGIT_RUN = re.compile(r"\d+")


def _indic_blocks_present(text: str) -> frozenset:
    """Set of Indic script blocks that appear in `text` (Latin/digits ignored)."""
    found = set()
    for ch in text:
        cp = ord(ch)
        for name, (lo, hi) in _INDIC_BLOCKS.items():
            if lo <= cp <= hi:
                found.add(name)
                break
    return frozenset(found)


def _passes_guards(original: str, corrected: str, segment: dict) -> bool:
    """Deterministic second line of defense, independent of the LLM prompt.
    Any failure -> reject the correction and keep the original text."""
    corrected = corrected or ""

    # Empty guard: never blank a non-empty transcript.
    if original.strip() and not corrected.strip():
        return False

    # Script guard: forbid translation / script conversion. The set of Indic
    # blocks must be identical (code-mixed Latin/digits are not counted, so
    # Telugu+English stays allowed; Telugu->Hindi or Telugu->Latin is rejected).
    if _indic_blocks_present(original) != _indic_blocks_present(corrected):
        return False

    # Number guard: every digit run in the original must survive verbatim.
    for run in _DIGIT_RUN.findall(original):
        if run not in corrected:
            return False

    # Word-count guard: allow at most ±1 word absolute delta so fragment merges
    # (2 ASR fragments → 1 real word) and word completions pass, while multi-word
    # rewrites are blocked. Overlapping speech stays at 0 (no word change at all —
    # cross-talk segments are too ambiguous for reconstruction). The confidence
    # threshold (0.92) is the primary hallucination defence; this is the backstop.
    o_words = original.split()
    c_words = corrected.split()
    if segment.get("overlap"):
        if len(c_words) != len(o_words):
            return False
    else:
        if abs(len(c_words) - len(o_words)) > 1:
            return False

    return True


_SYSTEM_PROMPT = """You are a forensic transcript corrector for multilingual Indian-language audio evidence (Telugu, Hindi, English, Tamil, Kannada, Malayalam, Marathi, Bengali, Punjabi, Gujarati, Odia, Assamese, Urdu, Sanskrit, and any code-mixed combination).

Your task: repair ASR output so every token is a real, meaningful word in the detected language. Return JSON only.

YOU MAY correct:
- Missing or incorrect characters within a word (e.g. duplicate vowel matras, missing halant)
- Broken words that ASR split mid-character — rejoin them into the correct single word
- Incomplete word fragments: complete them to the phonetically closest real word in that language
- Common phonetic substitutions that produce a known word
- Wrong Unicode character / script-level recognition errors within the same language
- Two adjacent ASR fragments that together form one word — merge them

YOU MUST NEVER:
- Translate to another language or change the script
- Paraphrase, rewrite, summarize, or expand meaning
- Change numbers (phone, Aadhaar, PAN, account, currency, dates, times)
- Change speaker labels, timestamps, or segment boundaries
- Invent names, locations, organizations, events, or factual context not implied by the phonetics
- Replace a word that is already a valid word in that language with a different word
- Generate entirely new content when the segment is pure noise with no phonetic signal

RULES:
- Goal: every token in corrected_text must be a real word (or recognised proper noun / code-mix English) in the detected language.
- A single-syllable fragment like "మ", "ని", "క" standing alone is likely an ASR partial — complete it ONLY if you are highly confident of the full word from phonetic context; otherwise leave unchanged.
- Preserve the original language and any code-mixing exactly (e.g. Telugu + English stays Telugu + English).
- If overlap is true, be EXTREMELY conservative: only fix unambiguous single-character spelling errors.
- Word count may change by ±1 if two ASR fragments should be one word or one fragment expands to a real word — but never add entirely new words.
- If you are not highly certain, leave the token unchanged. Do not guess.
- If nothing needs correcting, return correction_status "unchanged" with corrected_text equal to the original.

USING TRANSCRIPT CONTEXT:
- The user message includes the full raw transcript of the recording as `transcript_context`.
- Use it to identify proper nouns, recurring words, speaker names, domain vocabulary, and the topic of the conversation.
- If a fragment in the current segment resembles a word that appears clearly elsewhere in the transcript, reconstruct it to that word.
- Never copy whole phrases from other segments into the current segment — only use context to guide word-level reconstruction.

Return ONLY this JSON object, no prose:
{"correction_status":"corrected"|"unchanged","correction_confidence":<0.0-1.0>,"original_text":"<input>","corrected_text":"<output>","changes":[{"original":"<token>","corrected":"<token>","type":"spelling"|"asr_error"|"script"|"word_completion"|"fragment_merge"}]}"""


def _build_user_prompt(segment: dict) -> str:
    payload = {
        "text": segment.get("text", ""),
        "language": segment.get("language"),
        "overlap": bool(segment.get("overlap")),
        "confidence": segment.get("confidence"),
    }
    ctx = segment.get("transcript_context", "")
    if ctx:
        payload["transcript_context"] = ctx
    return json.dumps(payload, ensure_ascii=False)


def _parse_llm_json(raw: str) -> dict:
    """Extract the JSON object from a model response (tolerates surrounding text)."""
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def _ollama_generate(url: str, model: str, segment: dict) -> dict:
    """One Ollama /api/generate call via stdlib urllib — no third-party dep."""
    import urllib.request

    body = json.dumps({
        "model": model,
        "system": _SYSTEM_PROMPT,
        "prompt": _build_user_prompt(segment),
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/generate",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=config.LLM_ENHANCEMENT_TIMEOUT_S) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return _parse_llm_json(payload.get("response", ""))


def _call_ollama(segment: dict) -> dict:
    """Primary Ollama; on any failure fall back to the configured fallback
    server/model so a down local 14B does not disable the layer. Raises only if
    both primary and fallback fail (caller maps that to correction_status=error)."""
    try:
        return _ollama_generate(config.LLM_OLLAMA_URL, config.LLM_MODEL, segment)
    except Exception as primary_exc:
        fb_url = getattr(config, "LLM_OLLAMA_FALLBACK_URL", "")
        fb_model = getattr(config, "LLM_FALLBACK_MODEL", "")
        if not fb_url or not fb_model:
            raise
        logger.warning("primary Ollama failed (%s); trying fallback %s/%s: %s",
                       config.LLM_OLLAMA_URL, fb_url, fb_model, primary_exc)
        return _ollama_generate(fb_url, fb_model, segment)


def _call_anthropic(segment: dict) -> dict:
    """Anthropic provider (optional, deploy-only). Lazy import keeps it off the
    laptop build path."""
    import anthropic

    client = anthropic.Anthropic(api_key=config.LLM_ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(segment)}],
        timeout=config.LLM_ENHANCEMENT_TIMEOUT_S,
    )
    text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
    return _parse_llm_json(text)


def _call_llm(segment: dict) -> dict:
    if config.LLM_PROVIDER == "ollama":
        return _call_ollama(segment)
    if config.LLM_PROVIDER == "anthropic":
        return _call_anthropic(segment)
    raise ValueError(f"unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def enhance_segment(segment: dict) -> dict:
    """Correct one segment's ASR text. Always returns a record; never raises.

    Returns keys: correction_status, correction_confidence, original_text,
    corrected_text, changes. `corrected_text` equals the original for every
    status except "corrected"."""
    original = segment.get("text", "") or ""
    unchanged = {
        "correction_status": "unchanged",
        "correction_confidence": 1.0,
        "original_text": original,
        "corrected_text": original,
        "changes": [],
    }

    if not config.LLM_ENHANCEMENT_ENABLED:
        return {**unchanged, "correction_status": "not_run"}
    if not original.strip():
        return {**unchanged, "correction_status": "skipped"}
    if (segment.get("confidence") or 0.0) < config.LLM_ENHANCEMENT_MIN_CONF:
        return {**unchanged, "correction_status": "skipped"}

    try:
        result = _call_llm(segment)
    except Exception as exc:
        logger.warning("LLM enhancement failed (provider=%s): %s",
                       config.LLM_PROVIDER, exc)
        return {**unchanged, "correction_status": "error"}

    status = result.get("correction_status")
    corrected = result.get("corrected_text", original) or ""
    conf = result.get("correction_confidence")

    # Model said no change, or echoed the original -> unchanged.
    if status != "corrected" or corrected.strip() == original.strip():
        return {**unchanged, "correction_status": "unchanged"}

    # Apply only confident corrections.
    if not isinstance(conf, (int, float)) or conf < config.LLM_CORRECTION_MIN_CONF:
        return {**unchanged, "correction_status": "unchanged"}

    # Deterministic guards independent of the prompt.
    if not _passes_guards(original, corrected, segment):
        return {**unchanged, "correction_status": "guard_rejected"}

    return {
        "correction_status": "corrected",
        "correction_confidence": round(float(conf), 3),
        "original_text": original,
        "corrected_text": corrected,
        "changes": result.get("changes") or [],
    }
