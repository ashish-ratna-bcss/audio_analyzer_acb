# Independent Multi-Model ASR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Whisper, IndicConformer, and SeamlessM4T run independently on one robustly-preprocessed clip, emit per-model transcripts, validate them together with multilingual embeddings, and fix the 5 quality defects.

**Architecture:** Pure/dependency-injected helpers (hallucination filter, language vote, cross-model consensus, transcript builders) are unit-tested with mocked model output on the laptop. `pipeline/tasks.py` L5 wires them: pre-sweep MMS-LID → file prior, per-unit robust preprocessing, three independent model calls, consensus, persist all candidates. L8 writes three per-model files + a validation report.

**Tech Stack:** Python 3.12, faster-whisper, transformers (IndicConformer/SeamlessM4T/MMS-LID), sentence-transformers (LaBSE), ffmpeg, Silero VAD, SQLAlchemy, Celery, pytest + unittest.mock.

## Global Constraints

- Branch: `feat/forensic-v2-pipeline`. Commit per task. Do NOT push or deploy until all unit tests green.
- Laptop has NO GPU models — every test mocks model services. No test may load a real model.
- Forensic invariant: a segment ALWAYS produces a DB row; a model failure becomes an empty flagged pass, never aborts the job.
- IndicConformer must NEVER silently fall back to Whisper. On unsupported language it abstains.
- Preserve existing reconcile count checks (`units == segments`) and the WORM/audit/manifest chain (untouched).
- Keep `final/{file}_certified_transcript.json` (consensus) so existing `/review` + `/certify` endpoints keep working.
- Co-author trailer on every commit: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Config additions

**Files:**
- Modify: `config.py` (append after line 142, end of Phase 4 block)
- Test: `tests/test_config_multimodel.py` (create)

**Interfaces:**
- Produces: `config.ALLOWED_LANGS: set[str]`, `config.LID_VOTE_MIN_CONF: float`, `config.NO_SPEECH_MAX: float`, `config.AGREEMENT_MIN: float`, `config.LOUDNORM_LUFS: float`, `config.GHOST_PHRASES: list[str]`. Also changes `config.GAP_WINDOW_S` default `20.0 → 10.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_multimodel.py
import config


def test_multimodel_config_defaults():
    assert config.ALLOWED_LANGS == set()           # empty = open auto
    assert config.LID_VOTE_MIN_CONF == 0.5
    assert config.NO_SPEECH_MAX == 0.6
    assert config.AGREEMENT_MIN == 0.6
    assert config.LOUDNORM_LUFS == -16.0
    assert config.GAP_WINDOW_S == 10.0
    assert isinstance(config.GHOST_PHRASES, list)
    assert "thank you" in [p.lower() for p in config.GHOST_PHRASES]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_multimodel.py -v`
Expected: FAIL (AttributeError / GAP_WINDOW_S == 20.0)

- [ ] **Step 3: Implement**

In `config.py` change line 122 default `"20.0"` to `"10.0"`, then append:

```python
# --- Phase: independent multi-model ASR + cross-model validation ---

# Allowed language ISO-639-1 set. Empty = open auto-detect (default). Set e.g.
# ALLOWED_LANGS=te,en,hi to constrain a known-language case and kill LID misroutes.
ALLOWED_LANGS = {c.strip() for c in os.getenv("ALLOWED_LANGS", "").split(",") if c.strip()}

# Min MMS-LID top-1 confidence for a clip's LID to count toward the file vote
# and to be trusted over the file prior.
LID_VOTE_MIN_CONF = float(os.getenv("LID_VOTE_MIN_CONF", "0.5"))

# Whisper no_speech_prob above this blanks the pass (true non-speech / silence).
NO_SPEECH_MAX = float(os.getenv("NO_SPEECH_MAX", "0.6"))

# Mean pairwise embedding cosine below this flags a segment for cross-model disagreement.
AGREEMENT_MIN = float(os.getenv("AGREEMENT_MIN", "0.6"))

# EBU R128 integrated loudness target for per-clip normalization (LUFS).
LOUDNORM_LUFS = float(os.getenv("LOUDNORM_LUFS", "-16.0"))

# Known ASR hallucination phrases emitted on non-speech (training-data ghosts).
# Matched case- and punctuation-insensitive. Env-extendable via GHOST_PHRASES_EXTRA (comma-sep).
GHOST_PHRASES = [
    "thank you", "thank you.", "thanks for watching", "thanks for watching!",
    "please subscribe", "subscribe", "like and subscribe", ". .", "...",
    "[music]", "[music playing]", "[applause]", "(music)",
    "ご視聴ありがとうございました", "Продолжение следует...",
] + [p.strip() for p in os.getenv("GHOST_PHRASES_EXTRA", "").split(",") if p.strip()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_multimodel.py tests/test_config_phase4.py -v`
Expected: PASS (and existing phase4 config test still green)

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config_multimodel.py
git commit -m "feat: config for multi-model ASR (allowed langs, gates, ghost phrases)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Hallucination filter

**Files:**
- Create: `services/hallucination_filter.py`
- Test: `tests/test_hallucination_filter.py` (create)

**Interfaces:**
- Produces:
  - `has_repetition(text: str) -> bool` (moved from `pipeline/tasks.py:_has_repetition`)
  - `filter_pass(result: dict, *, no_speech_prob: float | None = None) -> dict` — returns the input dict unchanged, or with `text=""`, `confidence=0.0`, and a `hallucination` reason key when a no-speech / ghost / repetition hit occurs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hallucination_filter.py
from services.hallucination_filter import has_repetition, filter_pass


def test_repetition_consecutive():
    assert has_repetition("go go go now") is True

def test_repetition_low_unique_ratio():
    assert has_repetition("a a a a b a a a") is True

def test_no_repetition_normal():
    assert has_repetition("the cost is fifteen rupees") is False

def test_ghost_phrase_blanked():
    out = filter_pass({"text": "Thank you.", "confidence": 0.6})
    assert out["text"] == ""
    assert out["confidence"] == 0.0
    assert out["hallucination"] == "ghost_phrase"

def test_no_speech_blanked():
    out = filter_pass({"text": "real words here", "confidence": 0.7}, no_speech_prob=0.9)
    assert out["text"] == ""
    assert out["hallucination"] == "no_speech"

def test_repetition_blanked():
    out = filter_pass({"text": "fifteen fifteen fifteen fifteen", "confidence": 0.5})
    assert out["text"] == ""
    assert out["hallucination"] == "repetition"

def test_clean_passes_through():
    src = {"text": "the cost is fifteen rupees", "confidence": 0.8}
    out = filter_pass(src, no_speech_prob=0.1)
    assert out == src
    assert "hallucination" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hallucination_filter.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement**

```python
# services/hallucination_filter.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hallucination_filter.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add services/hallucination_filter.py tests/test_hallucination_filter.py
git commit -m "feat: hallucination_filter (no-speech, ghost-phrase, repetition gates)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: File-level language vote

**Files:**
- Modify: `services/lang_id_service.py` (append function)
- Test: `tests/test_lang_vote.py` (create)

**Interfaces:**
- Consumes: existing `to_iso639_1(mms_code)`.
- Produces: `vote_file_language(per_clip_lids: list[dict], *, allowed_langs: set[str], min_conf: float) -> str | None`. Each item is `{"top1": <iso639-3 or None>, "top1_confidence": float}`. Returns the majority ISO-639-1 prior or `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lang_vote.py
from services.lang_id_service import vote_file_language


def test_majority_vote():
    lids = [
        {"top1": "tel", "top1_confidence": 0.9},
        {"top1": "tel", "top1_confidence": 0.8},
        {"top1": "eng", "top1_confidence": 0.7},
    ]
    assert vote_file_language(lids, allowed_langs=set(), min_conf=0.5) == "te"

def test_low_conf_excluded():
    lids = [
        {"top1": "kor", "top1_confidence": 0.2},
        {"top1": "tel", "top1_confidence": 0.9},
    ]
    assert vote_file_language(lids, allowed_langs=set(), min_conf=0.5) == "te"

def test_allowed_set_filters():
    lids = [
        {"top1": "kor", "top1_confidence": 0.9},
        {"top1": "kor", "top1_confidence": 0.9},
        {"top1": "tel", "top1_confidence": 0.8},
    ]
    # kor not in allowed -> te wins despite fewer votes
    assert vote_file_language(lids, allowed_langs={"te", "en", "hi"}, min_conf=0.5) == "te"

def test_all_low_conf_returns_none():
    lids = [{"top1": "tel", "top1_confidence": 0.1}]
    assert vote_file_language(lids, allowed_langs=set(), min_conf=0.5) is None

def test_empty_returns_none():
    assert vote_file_language([], allowed_langs=set(), min_conf=0.5) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lang_vote.py -v`
Expected: FAIL (cannot import name 'vote_file_language')

- [ ] **Step 3: Implement** — append to `services/lang_id_service.py`:

```python
from collections import Counter


def vote_file_language(per_clip_lids, *, allowed_langs, min_conf):
    """Majority-vote a file-level language prior (ISO 639-1) from per-clip MMS-LID.

    Only clips with top1_confidence >= min_conf count. If allowed_langs is
    non-empty, only languages in it are eligible. Returns None if no eligible vote.
    """
    counts = Counter()
    for lid in per_clip_lids:
        if (lid.get("top1_confidence") or 0.0) < min_conf:
            continue
        iso1 = to_iso639_1(lid.get("top1"))
        if not iso1:
            continue
        if allowed_langs and iso1 not in allowed_langs:
            continue
        counts[iso1] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lang_vote.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add services/lang_id_service.py tests/test_lang_vote.py
git commit -m "feat: file-level language majority vote with confidence gate + allowed set

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Cross-model consensus (rewrite)

**Files:**
- Rewrite: `services/cross_model.py`
- Rewrite: `tests/test_cross_model.py` (old signature removed)

**Interfaces:**
- Produces: `compare_passes(texts: dict[str,str], confidences: dict[str,float|None], *, embed_fn, agreement_min: float = 0.6, conf_thresh: float = 0.5) -> dict`.
  `embed_fn(a: str, b: str) -> float` returns cosine similarity in [0,1]. Result dict keys:
  `consensus_pass: str|None`, `consensus_text: str`, `agreement: float`, `confidence: float`,
  `flagged: bool`, `flag_reason: str|None` (`insufficient_passes|cross_model_disagreement|low_confidence|None`).
- `normalized_edit_distance` is retained (still used by tests / utility).

- [ ] **Step 1: Write the failing test** — replace entire `tests/test_cross_model.py`:

```python
from services.cross_model import normalized_edit_distance, compare_passes


def _stub_embed(score):
    return lambda a, b: score


def test_edit_distance_bounds():
    assert normalized_edit_distance("abc", "abc") == 0.0
    assert normalized_edit_distance("abc", "xyz") == 1.0


def test_high_agreement_not_flagged():
    r = compare_passes(
        {"whisper": "the cost is fifteen", "indic": "ధర పదిహేను", "seamless": "the cost is fifteen"},
        {"whisper": 0.9, "indic": None, "seamless": 0.8},
        embed_fn=_stub_embed(0.92))
    assert r["flagged"] is False
    assert r["agreement"] >= 0.9
    assert r["consensus_pass"] in {"whisper", "indic", "seamless"}
    assert r["consensus_text"]


def test_insufficient_passes_flagged():
    r = compare_passes(
        {"whisper": "", "indic": "", "seamless": "something"},
        {"whisper": 0.0, "indic": None, "seamless": 0.7},
        embed_fn=_stub_embed(0.9))
    assert r["flagged"] is True
    assert r["flag_reason"] == "insufficient_passes"
    assert r["consensus_pass"] == "seamless"   # the only non-empty pass


def test_disagreement_flagged():
    r = compare_passes(
        {"whisper": "the cost is fifty", "indic": "totally different", "seamless": "unrelated words"},
        {"whisper": 0.8, "indic": 0.8, "seamless": 0.8},
        embed_fn=_stub_embed(0.2))
    assert r["flagged"] is True
    assert r["flag_reason"] == "cross_model_disagreement"


def test_low_confidence_flagged():
    r = compare_passes(
        {"whisper": "same text", "indic": "same text", "seamless": "same text"},
        {"whisper": 0.2, "indic": 0.2, "seamless": 0.2},
        embed_fn=_stub_embed(0.95))
    assert r["flagged"] is True
    assert r["flag_reason"] == "low_confidence"


def test_all_empty():
    r = compare_passes(
        {"whisper": "", "indic": "", "seamless": ""},
        {"whisper": 0.0, "indic": None, "seamless": 0.0},
        embed_fn=_stub_embed(0.0))
    assert r["flagged"] is True
    assert r["flag_reason"] == "insufficient_passes"
    assert r["consensus_text"] == ""


def test_embed_failure_degrades_not_crashes():
    def boom(a, b):
        raise RuntimeError("embed down")
    r = compare_passes(
        {"whisper": "a", "indic": "b", "seamless": "c"},
        {"whisper": 0.9, "indic": 0.9, "seamless": 0.9},
        embed_fn=boom)
    assert r["flagged"] is True
    assert r["agreement"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cross_model.py -v`
Expected: FAIL (signature/keys changed)

- [ ] **Step 3: Implement** — replace `services/cross_model.py`:

```python
def normalized_edit_distance(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 1.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] / max(la, lb)


def _pairwise_agreement(texts, embed_fn):
    """Mean pairwise cosine over the given (key,text) pairs, plus per-key mean sim."""
    keys = list(texts)
    if len(keys) < 2:
        return 1.0 if keys else 0.0, {k: 1.0 for k in keys}
    sims = {k: [] for k in keys}
    total, n = 0.0, 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            s = float(embed_fn(texts[keys[i]], texts[keys[j]]))
            sims[keys[i]].append(s)
            sims[keys[j]].append(s)
            total += s
            n += 1
    mean = total / n if n else 0.0
    per_key = {k: (sum(v) / len(v) if v else 0.0) for k, v in sims.items()}
    return mean, per_key


def compare_passes(texts, confidences, *, embed_fn, agreement_min=0.6, conf_thresh=0.5):
    """Validate the 3 ASR passes together. Script-agnostic (embeddings).

    Returns consensus pick + agreement + flag. Never raises: embedding failure
    degrades to agreement 0.0 + flagged.
    """
    nonempty = {k: v.strip() for k, v in texts.items() if v and v.strip()}

    # Confidence mean over real (non-None) scores only.
    real_confs = [c for c in confidences.values() if isinstance(c, (int, float))]
    mean_conf = round(sum(real_confs) / len(real_confs), 3) if real_confs else 0.0

    if len(nonempty) < 2:
        only = next(iter(nonempty), None)
        return {"consensus_pass": only,
                "consensus_text": nonempty.get(only, "") if only else "",
                "agreement": 0.0, "confidence": mean_conf,
                "flagged": True, "flag_reason": "insufficient_passes"}

    try:
        agreement, per_key = _pairwise_agreement(nonempty, embed_fn)
        embed_ok = True
    except Exception:
        agreement, per_key, embed_ok = 0.0, {k: 0.0 for k in nonempty}, False

    # Consensus = medoid (highest mean similarity to peers); tie-break by confidence.
    def _score(k):
        return (per_key.get(k, 0.0), confidences.get(k) or 0.0)
    consensus_pass = max(nonempty, key=_score)
    consensus_text = nonempty[consensus_pass]

    if not embed_ok or agreement < agreement_min:
        return {"consensus_pass": consensus_pass, "consensus_text": consensus_text,
                "agreement": round(agreement, 3), "confidence": mean_conf,
                "flagged": True, "flag_reason": "cross_model_disagreement"}
    if mean_conf < conf_thresh:
        return {"consensus_pass": consensus_pass, "consensus_text": consensus_text,
                "agreement": round(agreement, 3), "confidence": mean_conf,
                "flagged": True, "flag_reason": "low_confidence"}
    return {"consensus_pass": consensus_pass, "consensus_text": consensus_text,
            "agreement": round(agreement, 3), "confidence": mean_conf,
            "flagged": False, "flag_reason": None}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cross_model.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add services/cross_model.py tests/test_cross_model.py
git commit -m "feat: rewrite cross_model — multilingual embedding consensus, script-agnostic

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: IndicConformer — abstain + honest confidence

**Files:**
- Modify: `services/indic_asr_service.py:40-99` (`transcribe_clip`)
- Test: `tests/test_indic_abstain.py` (create)

**Interfaces:**
- Produces (changed contract): `transcribe_clip(wav_path, lang_code) -> dict` with keys
  `text`, `confidence` (`float` when the model ran and produced text, else `None`), `language`,
  `model`, and `abstained: bool`. For `lang_code ∉ _INDIC_SUPPORTED`: returns
  `{"text": "", "confidence": None, "language": lang_code, "model": "indic_unsupported",
  "abstained": True}` — NO Whisper fallback.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_indic_abstain.py
from unittest.mock import patch
from services import indic_asr_service as ias


def test_abstains_on_non_indic():
    r = ias.transcribe_clip("/tmp/x.wav", "ko")
    assert r["abstained"] is True
    assert r["text"] == ""
    assert r["confidence"] is None
    assert r["model"] == "indic_unsupported"


def test_runs_for_indic_and_returns_text():
    class _M:
        def __call__(self, wav, lang, mode): return ["పదిహేను"]
        def eval(self): return self
        def to(self, d): return self
    with patch.object(ias, "_load", return_value=_M()), \
         patch("torchaudio.load", return_value=(__import__("torch").zeros(1, 16000), 16000)):
        r = ias.transcribe_clip("/tmp/x.wav", "te")
    assert r["abstained"] is False
    assert r["text"] == "పదిహేను"
    assert r["confidence"] is None        # CTC checkpoint exposes no score -> unscored, not faked
    assert r["language"] == "te"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_indic_abstain.py -v`
Expected: FAIL (KeyError 'abstained' / falls back to whisper)

- [ ] **Step 3: Implement** — replace the body of `transcribe_clip` in `services/indic_asr_service.py`:

```python
def transcribe_clip(wav_path: str, lang_code: str) -> dict:
    """Pass 2 ASR — IndicConformer-600M only. Independent of Whisper.

    On a language IndicConformer does not cover, ABSTAIN (empty + flag). Never
    silently fall back to Whisper — that would break model independence.
    Confidence is None ("unscored"): the CTC checkpoint exposes no score, and a
    fabricated constant must not enter the forensic record.
    """
    if lang_code not in _INDIC_SUPPORTED:
        return {"text": "", "confidence": None, "language": lang_code,
                "model": "indic_unsupported", "abstained": True}
    try:
        import torch
        import torchaudio

        model = _load()
        wav, sr = torchaudio.load(wav_path)
        wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        if config.WHISPER_DEVICE == "cuda":
            wav = wav.to("cuda")
        with torch.jit.fuser("none"):
            result = model(wav, lang_code, "ctc")
        if isinstance(result, (list, tuple)):
            text = (result[0] if result else "") or ""
        else:
            text = result or ""
        text = str(text).strip()
        return {"text": text, "confidence": None, "language": lang_code,
                "model": config.INDIC_CONFORMER_MODEL, "abstained": False}
    except Exception as exc:
        logger.warning("IndicConformer failed for %s (lang=%s): %s", wav_path, lang_code, exc)
        return {"text": "", "confidence": None, "language": lang_code,
                "model": "indic_error", "abstained": True}
```

(Remove the old Whisper-fallback tail and the `model_used`/`confidence = 0.75` lines.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_indic_abstain.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add services/indic_asr_service.py tests/test_indic_abstain.py
git commit -m "fix: IndicConformer abstains on unsupported lang, no fake conf, no whisper masquerade

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Robust preprocessing service

**Files:**
- Create: `services/preprocess_service.py`
- Test: `tests/test_preprocess_service.py` (create)

**Interfaces:**
- Consumes: `clip_service.cut`, `vad_service.detect_speech`.
- Produces: `prepare_clip(enhanced_src, original_src, start, end, workdir, idx, speaker) -> dict`
  returning `{"clean": <path>, "raw": <path>}`. `clean` = enhanced cut → loudnorm → edge-trim;
  `raw` = original cut (unprocessed, audit). Helper `_loudnorm(in_path, out_path) -> bool`
  (False on ffmpeg failure → caller keeps pre-loudnorm clip).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocess_service.py
from unittest.mock import patch, MagicMock
from services import preprocess_service as pp


def test_prepare_clip_pipeline_wires_steps(tmp_path):
    calls = {"cut": 0, "loud": 0, "vad": 0}

    def fake_cut(src, s, e, out, normalize=False):
        calls["cut"] += 1
        open(out, "wb").close()
        return out

    def fake_loud(in_p, out_p):
        calls["loud"] += 1
        open(out_p, "wb").close()
        return True

    def fake_vad(path):
        calls["vad"] += 1
        return [{"start": 0.2, "end": 1.8}]   # trim edges to detected speech

    with patch.object(pp.clip_service, "cut", side_effect=fake_cut), \
         patch.object(pp, "_loudnorm", side_effect=fake_loud), \
         patch.object(pp.vad_service, "detect_speech", side_effect=fake_vad):
        out = pp.prepare_clip(str(tmp_path / "enh.wav"), str(tmp_path / "org.wav"),
                              10.0, 13.0, str(tmp_path), 0, "Speaker_1")

    assert "clean" in out and "raw" in out
    assert calls["loud"] == 1
    assert calls["vad"] >= 1
    assert calls["cut"] >= 2   # at least raw + enhanced cuts


def test_loudnorm_failure_keeps_clip(tmp_path):
    def fake_cut(src, s, e, out, normalize=False):
        open(out, "wb").close()
        return out

    with patch.object(pp.clip_service, "cut", side_effect=fake_cut), \
         patch.object(pp, "_loudnorm", return_value=False), \
         patch.object(pp.vad_service, "detect_speech", return_value=[]):
        out = pp.prepare_clip(str(tmp_path / "enh.wav"), str(tmp_path / "org.wav"),
                              0.0, 2.0, str(tmp_path), 1, "Speaker_2")
    # No crash; clean path still returned (falls back to pre-loudnorm cut).
    assert out["clean"]
    assert out["raw"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preprocess_service.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement**

```python
# services/preprocess_service.py
"""Uniform robust preprocessing: one clean clip per unit, fed to all 3 ASR models.

Chain: cut from the DeepFilterNet3-enhanced full file -> EBU R128 loudness
normalize -> trim leading/trailing silence (never cut interior). The raw cut
from the ORIGINAL file is always kept unprocessed for audit. Every step is
non-fatal: a failure falls back to the previous clip, never raises.
"""
import os
import subprocess

import config
from services import clip_service, vad_service


def _loudnorm(in_path: str, out_path: str) -> bool:
    """EBU R128 loudness normalization. Returns False on ffmpeg failure."""
    cmd = ["ffmpeg", "-y", "-i", in_path,
           "-af", f"loudnorm=I={config.LOUDNORM_LUFS}:TP=-1.5:LRA=11",
           "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0 and os.path.exists(out_path)
    except Exception:
        return False


def prepare_clip(enhanced_src, original_src, start, end, workdir, idx, speaker):
    """Produce {'clean': path, 'raw': path} for one transcription unit."""
    raw = os.path.join(workdir, f"seg_{idx:04d}_{speaker}_org.wav")
    clip_service.cut(original_src, start, end, raw, normalize=False)

    enh_cut = os.path.join(workdir, f"seg_{idx:04d}_{speaker}_enh.wav")
    clip_service.cut(enhanced_src, start, end, enh_cut, normalize=False)

    # Loudness normalize (fall back to enh_cut if it fails).
    loud = os.path.join(workdir, f"seg_{idx:04d}_{speaker}_loud.wav")
    clean = loud if _loudnorm(enh_cut, loud) else enh_cut

    # Trim leading/trailing silence to detected speech span (keep interior).
    speech = vad_service.detect_speech(clean)
    if speech:
        s0 = min(seg["start"] for seg in speech)
        e0 = max(seg["end"] for seg in speech)
        if e0 - s0 > 0.1:
            trimmed = os.path.join(workdir, f"seg_{idx:04d}_{speaker}_clean.wav")
            try:
                clip_service.cut(clean, s0, e0, trimmed, normalize=False)
                clean = trimmed
            except Exception:
                pass  # keep untrimmed clean clip

    return {"clean": clean, "raw": raw}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_preprocess_service.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add services/preprocess_service.py tests/test_preprocess_service.py
git commit -m "feat: preprocess_service — uniform denoise+loudnorm+trim clip for all models

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Per-model transcript + validation report builders

**Files:**
- Modify: `services/transcript_service.py` (append builders + named writer)
- Test: `tests/test_transcript_outputs.py` (create)

**Interfaces:**
- Consumes: persisted segment objects exposing `.id .start .end .speaker .detected_language
  .flagged` and `.candidates` (dict with `pass1_whisper|pass2_indic_conformer|pass3_seamless`
  sub-dicts of `{text, confidence}`, plus `agreement`, `consensus_pass`).
- Produces:
  - `build_per_model(file_id, segments, pass_key) -> dict`
  - `build_validation_report(file_id, segments) -> dict`
  - `write_named(case_id, file_id, name, data) -> str` (writes `final/{file}_{name}.json`)
  - `PASS_FILE_NAMES = {"pass1_whisper": "whisper_transcript", ...}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transcript_outputs.py
from types import SimpleNamespace
from services import transcript_service as ts


def _seg():
    return SimpleNamespace(
        id="s1", start=1.0, end=2.0, speaker="Speaker_1",
        detected_language="te", flagged=True,
        candidates={
            "pass1_whisper": {"text": "hello", "confidence": 0.8},
            "pass2_indic_conformer": {"text": "హలో", "confidence": None},
            "pass3_seamless": {"text": "", "confidence": 0.0, "hallucination": "no_speech"},
            "agreement": 0.71, "consensus_pass": "pass1_whisper",
        })


def test_build_per_model_whisper():
    d = ts.build_per_model("f1", [_seg()], "pass1_whisper")
    assert d["file_id"] == "f1"
    assert d["model"] == "pass1_whisper"
    seg = d["segments"][0]
    assert seg["text"] == "hello"
    assert seg["confidence"] == 0.8
    assert seg["speaker"] == "Speaker_1"


def test_build_per_model_seamless_empty_keeps_flag():
    d = ts.build_per_model("f1", [_seg()], "pass3_seamless")
    seg = d["segments"][0]
    assert seg["text"] == ""
    assert seg["flagged_for_review"] is True


def test_validation_report_shape():
    r = ts.build_validation_report("f1", [_seg()])
    assert r["file_id"] == "f1"
    seg = r["segments"][0]
    assert seg["whisper"]["text"] == "hello"
    assert seg["indic"]["text"] == "హలో"
    assert seg["agreement"] == 0.71
    assert seg["consensus_pass"] == "pass1_whisper"
    assert "summary" in r
    assert r["summary"]["segments_total"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcript_outputs.py -v`
Expected: FAIL (no build_per_model)

- [ ] **Step 3: Implement** — append to `services/transcript_service.py`:

```python
PASS_FILE_NAMES = {
    "pass1_whisper": "whisper_transcript",
    "pass2_indic_conformer": "indic_transcript",
    "pass3_seamless": "seamless_transcript",
}


def build_per_model(file_id, segments, pass_key) -> dict:
    """Independent transcript for one model from persisted segment candidates."""
    out = []
    for s in segments:
        cand = (s.candidates or {}).get(pass_key) or {}
        out.append({
            "segment_id": s.id, "start": s.start, "end": s.end,
            "speaker": s.speaker, "language": s.detected_language,
            "text": cand.get("text", ""), "confidence": cand.get("confidence"),
            "hallucination": cand.get("hallucination"),
            "flagged_for_review": bool(s.flagged),
        })
    return {"file_id": file_id, "model": pass_key, "segments": out}


def build_validation_report(file_id, segments) -> dict:
    out, flags = [], {}
    for s in segments:
        c = s.candidates or {}
        w = c.get("pass1_whisper") or {}
        i = c.get("pass2_indic_conformer") or {}
        m = c.get("pass3_seamless") or {}
        seg_flags = [v.get("hallucination") for v in (w, i, m) if v.get("hallucination")]
        for fr in seg_flags:
            flags[fr] = flags.get(fr, 0) + 1
        out.append({
            "segment_id": s.id, "start": s.start, "end": s.end, "speaker": s.speaker,
            "whisper": {"text": w.get("text", ""), "confidence": w.get("confidence")},
            "indic": {"text": i.get("text", ""), "confidence": i.get("confidence")},
            "seamless": {"text": m.get("text", ""), "confidence": m.get("confidence")},
            "agreement": c.get("agreement"), "consensus_pass": c.get("consensus_pass"),
            "flags": seg_flags,
        })
    agreements = [o["agreement"] for o in out if isinstance(o["agreement"], (int, float))]
    summary = {
        "segments_total": len(out),
        "mean_agreement": round(sum(agreements) / len(agreements), 3) if agreements else 0.0,
        "hallucination_flags": flags,
    }
    return {"file_id": file_id, "segments": out, "summary": summary}


def write_named(case_id, file_id, name, data) -> str:
    path = storage.derivative_path(case_id, file_id, "final", f"{file_id}_{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transcript_outputs.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add services/transcript_service.py tests/test_transcript_outputs.py
git commit -m "feat: per-model transcript + validation report builders

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Pipeline integration — run_models, pre-sweep, consensus, L8

**Files:**
- Modify: `pipeline/tasks.py` — replace `_whisper_clip`/`_has_repetition`/`_three_pass_asr`/`_emit_segment`/`_l5_l6_segments`; extend L8 block in `run_pipeline`.
- Test: `tests/test_pipeline_attribution.py` (extend/replace relevant tests)

**Interfaces:**
- Consumes: Tasks 2–7 (`hallucination_filter.filter_pass`, `lang_id_service.{identify,vote_file_language,to_iso639_1}`, `cross_model.compare_passes`, `preprocess_service.prepare_clip`, `indic_asr_service.transcribe_clip`, `transcript_service.{build_per_model,build_validation_report,write_named,PASS_FILE_NAMES}`, `embedding_service.similarity`).
- Produces: `run_models(clean_clip, *, file_prior) -> dict` (keys `lang_id`, `whisper`, `indic`, `seamless`); rewired `_emit_segment` storing all candidates + consensus, `source_pass=consensus_pass` (fixes D1).

- [ ] **Step 1: Write the failing test** — add to `tests/test_pipeline_attribution.py`:

```python
from unittest.mock import patch
from pipeline import tasks


def test_run_models_independent_indic_abstains_non_indic():
    with patch.object(tasks.lang_id_service, "identify",
                      return_value={"top1": "kor", "top1_confidence": 0.9,
                                    "top2": None, "top2_confidence": 0.0}), \
         patch.object(tasks.lang_id_service, "to_iso639_1", return_value="ko"), \
         patch.object(tasks, "_whisper_clip",
                      return_value={"text": "annyeong", "confidence": 0.7,
                                    "language": "ko", "no_speech_prob": 0.1}), \
         patch.object(tasks.indic_asr_service, "transcribe_clip",
                      return_value={"text": "", "confidence": None, "language": "ko",
                                    "model": "indic_unsupported", "abstained": True}), \
         patch.object(tasks.seamless_service, "transcribe_clip",
                      return_value={"text": "annyeong", "confidence": 0.6, "language": "ko"}):
        asr = tasks.run_models("/tmp/clean.wav", file_prior=None)

    assert asr["indic"]["abstained"] is True
    assert asr["indic"]["text"] == ""          # never masquerades as whisper
    assert asr["whisper"]["text"] == "annyeong"


def test_run_models_blanks_ghost_phrase():
    with patch.object(tasks.lang_id_service, "identify",
                      return_value={"top1": "eng", "top1_confidence": 0.2,
                                    "top2": None, "top2_confidence": 0.0}), \
         patch.object(tasks.lang_id_service, "to_iso639_1", return_value="en"), \
         patch.object(tasks, "_whisper_clip",
                      return_value={"text": "Thank you.", "confidence": 0.6,
                                    "language": "en", "no_speech_prob": 0.2}), \
         patch.object(tasks.indic_asr_service, "transcribe_clip",
                      return_value={"text": "Thank you.", "confidence": None, "language": "en",
                                    "model": "x", "abstained": False}), \
         patch.object(tasks.seamless_service, "transcribe_clip",
                      return_value={"text": "[Music playing]", "confidence": 0.3, "language": "en"}):
        asr = tasks.run_models("/tmp/clean.wav", file_prior=None)

    assert asr["whisper"]["text"] == ""        # ghost blanked
    assert asr["seamless"]["text"] == ""        # [Music playing] blanked
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_attribution.py -v`
Expected: FAIL (no attribute `run_models`)

- [ ] **Step 3: Implement** — in `pipeline/tasks.py`:

(a) Update imports near line 13-19 to add:
```python
from services import preprocess_service, hallucination_filter
```

(b) Replace `_whisper_clip` (lines 152-159) to also surface `no_speech_prob`:
```python
def _whisper_clip(clip_path, task, language="auto"):
    res = whisper_service.transcribe(clip_path, language=language, use_vad=False, task=task)
    segs = res["segments"]
    if not segs:
        return {"text": "", "confidence": 0.0, "language": res.get("language", "und"),
                "no_speech_prob": 1.0}
    nsp = sum(s.get("no_speech_prob", 0.0) for s in segs) / len(segs)
    return {"text": " ".join(s["text"] for s in segs).strip(),
            "confidence": round(sum(s["confidence"] for s in segs) / len(segs), 3),
            "language": res.get("language", "und"), "no_speech_prob": round(nsp, 3)}
```

(c) Delete `_has_repetition` (lines 162-176) and `_clean_pass` (179-188) — now in `hallucination_filter`.

(d) Replace `_three_pass_asr` (303-329) with `run_models`:
```python
def run_models(clean_clip, *, file_prior):
    """Three independent ASR models on one robustly-preprocessed clip.

    Language routing: trust clip MMS-LID above the gate (and within ALLOWED_LANGS
    when set), else the file prior, else Whisper self-detect. Each pass is passed
    through the hallucination filter. Models never substitute for one another.
    """
    mms = lang_id_service.identify(clean_clip)
    clip_lang = lang_id_service.to_iso639_1(mms.get("top1"))
    clip_conf = mms.get("top1_confidence") or 0.0

    if clip_lang and clip_conf >= config.LID_VOTE_MIN_CONF and (
            not config.ALLOWED_LANGS or clip_lang in config.ALLOWED_LANGS):
        routing_lang = clip_lang
    elif file_prior:
        routing_lang = file_prior
    else:
        routing_lang = None  # whisper auto

    # Whisper (independent; constrained only when we have a confident routing lang).
    w = _whisper_clip(clean_clip, "transcribe", language=routing_lang or "auto")
    whisper_lang = w.get("language") or "und"
    if routing_lang is None:
        routing_lang = whisper_lang
    w = hallucination_filter.filter_pass(w, no_speech_prob=w.get("no_speech_prob"))

    # IndicConformer (abstains on unsupported; never whisper fallback).
    i = indic_asr_service.transcribe_clip(clean_clip, routing_lang)
    i = hallucination_filter.filter_pass(i)

    # SeamlessM4T (independent).
    m = seamless_service.transcribe_clip(clean_clip, routing_lang)
    m = hallucination_filter.filter_pass(m)

    return {"lang_id": {"mms_top1": mms.get("top1"),
                        "mms_top1_confidence": mms.get("top1_confidence"),
                        "mms_top2": mms.get("top2"),
                        "mms_top2_confidence": mms.get("top2_confidence"),
                        "whisper_lang": whisper_lang, "routing_lang": routing_lang},
            "whisper": w, "indic": i, "seamless": m}
```

(e) Replace `_emit_segment` (332-388) to use consensus:
```python
def _emit_segment(job, session, *, start, end, speaker, asr, clip_clean, clip_raw,
                  diarization_meta, extra_flags):
    w, i, m = asr["whisper"], asr["indic"], asr["seamless"]
    candidates = {
        "lang_id": asr["lang_id"],
        "pass1_whisper": {"text": w["text"], "confidence": w.get("confidence"),
                          "language": asr["lang_id"]["whisper_lang"],
                          "model": "openai/whisper-large-v3",
                          "hallucination": w.get("hallucination")},
        "pass2_indic_conformer": {"text": i["text"], "confidence": i.get("confidence"),
                                  "language": asr["lang_id"]["routing_lang"],
                                  "model": i.get("model", config.INDIC_CONFORMER_MODEL),
                                  "abstained": i.get("abstained", False),
                                  "hallucination": i.get("hallucination")},
        "pass3_seamless": {"text": m["text"], "confidence": m.get("confidence"),
                           "language": asr["lang_id"]["routing_lang"],
                           "model": config.SEAMLESS_MODEL,
                           "hallucination": m.get("hallucination")},
        "diarization": diarization_meta,
    }
    texts = {"pass1_whisper": w["text"], "pass2_indic_conformer": i["text"],
             "pass3_seamless": m["text"]}
    confs = {"pass1_whisper": w.get("confidence"), "pass2_indic_conformer": i.get("confidence"),
             "pass3_seamless": m.get("confidence")}
    verdict = cross_model.compare_passes(texts, confs, embed_fn=embedding_service.similarity,
                                         agreement_min=config.AGREEMENT_MIN)
    candidates["agreement"] = verdict["agreement"]
    candidates["consensus_pass"] = verdict["consensus_pass"]

    reasons = []
    if verdict["flagged"] and verdict.get("flag_reason"):
        reasons.append(verdict["flag_reason"])
    reasons.extend(extra_flags or [])
    flagged = bool(reasons)
    flag_reason = "+".join(dict.fromkeys(reasons)) if reasons else None

    seg_id = repo.add_segment(
        session, file_id=job.file_id, start=start, end=end, speaker=speaker,
        text=verdict["consensus_text"], confidence=verdict["confidence"],
        source_pass=verdict["consensus_pass"] or "none", flagged=flagged,
        review_status="pending" if flagged else "auto_accepted",
        candidates=candidates, clip_original=clip_raw, clip_enhanced=clip_clean,
        detected_language=asr["lang_id"]["routing_lang"])
    entry = {"segment_id": seg_id, "edit_distance_norm": None,
             "embedding_similarity": verdict["agreement"], "avg_logprob": None,
             "flag_reason": flag_reason}
    return seg_id, flagged, entry
```

(f) Rewrite `_l5_l6_segments` (391-460) to pre-sweep LID, use `prepare_clip` + `run_models`:
```python
def _l5_l6_segments(job, union, turns, enhanced16, original16, session):
    workdir = os.path.dirname(
        storage.derivative_path(job.case_id, job.file_id, "clips", "_"))
    per_segment, flagged_count = [], 0
    enh_source = enhanced16 or original16
    units = _build_units(turns, union)

    import torch as _torch

    # Pre-sweep: cheap MMS-LID over every unit clip -> file-level language prior.
    prepared = []
    lids = []
    for idx, unit in enumerate(units):
        clips = preprocess_service.prepare_clip(
            enh_source, original16, unit["start"], unit["end"], workdir, idx, unit["speaker"])
        prepared.append(clips)
        lids.append(lang_id_service.identify(clips["clean"]))
    file_prior = lang_id_service.vote_file_language(
        lids, allowed_langs=config.ALLOWED_LANGS, min_conf=config.LID_VOTE_MIN_CONF)

    for idx, unit in enumerate(units):
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        clips = prepared[idx]
        utype = unit["type"]
        speaker = unit["speaker"]

        if (utype == "overlap" and config.OVERLAP_SEPARATION_ENABLED
                and (unit["end"] - unit["start"]) >= config.OVERLAP_MIN_DUR_S):
            stems = separation_service.separate_speakers(
                clips["clean"], workdir, f"seg_{idx:04d}")
            if stems:
                for si, stem in enumerate(stems):
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                    spk = (unit["speakers"][si] if si < len(unit["speakers"])
                           else f"overlap_spk{si}")
                    asr = run_models(stem, file_prior=file_prior)
                    meta = {"speaker": spk,
                            "concurrent_speakers": [s for s in unit["speakers"] if s != spk],
                            "is_overlap": True, "segment_type": "overlap",
                            "separation": "sepformer", "stem_index": si}
                    _, flagged, entry = _emit_segment(
                        job, session, start=unit["start"], end=unit["end"], speaker=spk,
                        asr=asr, clip_clean=stem, clip_raw=stem,
                        diarization_meta=meta, extra_flags=["overlapping_speech"])
                    flagged_count += 1 if flagged else 0
                    per_segment.append(entry)
                continue

        asr = run_models(clips["clean"], file_prior=file_prior)
        extra = []
        if utype == "overlap":
            extra.append("overlapping_speech")
        if utype == "gap":
            extra.append("gap_recovery")
        meta = {"speaker": speaker,
                "concurrent_speakers": [s for s in unit["speakers"] if s != speaker],
                "is_overlap": utype == "overlap", "segment_type": utype}
        _, flagged, entry = _emit_segment(
            job, session, start=unit["start"], end=unit["end"], speaker=speaker,
            asr=asr, clip_clean=clips["clean"], clip_raw=clips["raw"],
            diarization_meta=meta, extra_flags=extra)
        flagged_count += 1 if flagged else 0
        per_segment.append(entry)

    session.commit()
    reconcile.check("L4:units", len(units), "L5:segments", len(per_segment))
    return per_segment, flagged_count
```

(g) Extend the L8 block in `run_pipeline` (after `ts.write(...)` near line 563) to also write per-model files + validation report:
```python
            for pass_key, name in ts.PASS_FILE_NAMES.items():
                ts.write_named(job.case_id, job.file_id, name,
                               ts.build_per_model(job.file_id, segs, pass_key))
            ts.write_named(job.case_id, job.file_id, "validation_report",
                           ts.build_validation_report(job.file_id, segs))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline_attribution.py tests/test_pipeline_skeleton.py tests/test_segment_candidates.py -v`
Expected: PASS (adjust any test referencing the removed `_three_pass_asr`/`source_pass=="pass1_whisper"` to the new consensus behavior)

- [ ] **Step 5: Commit**

```bash
git add pipeline/tasks.py tests/test_pipeline_attribution.py
git commit -m "feat: independent run_models + LID pre-sweep + consensus + per-model L8 output

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Full suite + deploy

**Files:** none (verification + ops)

- [ ] **Step 1: Run the whole suite**

Run: `pytest -q`
Expected: all pass. Fix any test that referenced removed symbols (`_clean_pass`, `_has_repetition`, old `compare_passes` signature, `source_pass=="pass1_whisper"`).

- [ ] **Step 2: Push**

```bash
git push origin feat/forensic-v2-pipeline
```

- [ ] **Step 3: Build + deploy on server**

```bash
ssh -i ~/Downloads/acb_processor.pem ubuntu@98.86.63.69 \
  'cd ~/app && git pull && docker compose build api && docker compose up -d'
```
Expected: containers healthy; `curl http://localhost/health` -> `{"status":"ok"}`.

- [ ] **Step 4: Real end-to-end validation**

Re-run the test video through the API. Confirm:
- three per-model transcript files written + validation_report.json;
- whisper / indic / seamless texts differ meaningfully (independence);
- silence regions (first ~140s of the test file) no longer emit ghost text;
- validation_report mean_agreement is plausible (not all 0).

---

## Self-Review

**Spec coverage:** D1 (Task 8e consensus source_pass), D2 (Task 2 + Task 8d wiring), D3 (Task 4), D4 (Tasks 1,3,8d), D5 (Task 5). Preprocess uniform (Task 6 + 8f). Per-model output + validation (Task 7 + 8g). Config (Task 1). Language vote (Task 3). All spec sections mapped.

**Placeholder scan:** every code step has full code; no TBD/TODO.

**Type consistency:** `prepare_clip` returns `{clean, raw}` (Task 6) — consumed as `clips["clean"]`/`clips["raw"]` (Task 8f). `compare_passes(..., embed_fn=...)` (Task 4) called with `embed_fn=embedding_service.similarity` (Task 8e). `transcribe_clip` returns `abstained` (Task 5) — read in Task 8e. `build_per_model`/`build_validation_report`/`write_named`/`PASS_FILE_NAMES` (Task 7) — used in Task 8g. Candidate keys `pass1_whisper|pass2_indic_conformer|pass3_seamless`, `agreement`, `consensus_pass` consistent across Tasks 7 & 8e.
