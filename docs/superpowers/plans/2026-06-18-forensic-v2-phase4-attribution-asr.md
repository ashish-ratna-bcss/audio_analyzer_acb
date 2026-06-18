# Forensic v2 — Phase 4 (Attribution + Multi-Pass ASR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Turn VAD-union regions into attributed, multiply-transcribed, confidence-scored segments — L4 overlap-aware diarization, L5 three-pass ASR (enhanced Whisper + original Whisper + Indic), L6 cross-model comparison/confidence/flagging — persisting `Segment` rows and the diarization/ASR/confidence JSON artifacts.

**Architecture:** Decision logic (edit-distance, confidence scoring, flag routing, overlap-aware speaker assignment) lives in pure modules fully unit-tested. Model wrappers (Indic ASR, multilingual embedding, pyannote overlap) are lazy + mocked in tests. The pipeline persists one `Segment` per union region with the winning pass + confidence + flag.

**Tech Stack:** faster-whisper, pyannote.audio, IndicConformer/IndicWhisper, sentence-transformers (LaBSE), torch.

## Global Constraints

- No ML model execution locally; wrappers lazy + gpu-marked; pipeline tests mock them.
- Flag a segment when: cross-model disagreement, OR low confidence, OR VAD-positive but any ASR pass empty (forced lowest confidence).
- Diarization never drops a region — at worst labels it `multi-speaker`.
- All prior tests stay green.

---

### Task 1: Cross-model comparison + confidence (pure)

**Files:**
- Create: `services/cross_model.py`
- Test: `tests/test_cross_model.py`

**Interfaces:**
- `normalized_edit_distance(a: str, b: str) -> float` (0 identical … 1 fully different).
- `compare_passes(texts: dict[str,str], confidences: dict[str,float], vad_positive: bool, embedding_sim: float | None, *, edit_thresh=0.4, sim_thresh=0.6, conf_thresh=0.5) -> dict` returns `{confidence, disagreement, flagged, flag_reason}`. `flag_reason ∈ {None,"vad_positive_asr_empty","cross_model_disagreement","low_logprob_confidence"}` (that priority).

- [ ] **Step 1: Failing test**

```python
# tests/test_cross_model.py
from services.cross_model import normalized_edit_distance, compare_passes


def test_edit_distance_bounds():
    assert normalized_edit_distance("abc", "abc") == 0.0
    assert normalized_edit_distance("", "") == 0.0
    assert normalized_edit_distance("abc", "xyz") == 1.0
    assert 0 < normalized_edit_distance("kitten", "sitting") < 1


def test_agreement_high_confidence():
    r = compare_passes(
        {"p1": "the cost is fifteen", "p2": "the cost is fifteen", "p3": "the cost is fifteen"},
        {"p1": 0.9, "p2": 0.85, "p3": 0.8}, vad_positive=True, embedding_sim=0.95)
    assert r["disagreement"] is False
    assert r["flagged"] is False
    assert r["flag_reason"] is None
    assert r["confidence"] > 0.8


def test_empty_pass_on_vad_positive_is_lowest():
    r = compare_passes(
        {"p1": "", "p2": "something", "p3": "something"},
        {"p1": 0.0, "p2": 0.7, "p3": 0.7}, vad_positive=True, embedding_sim=0.9)
    assert r["flagged"] is True
    assert r["flag_reason"] == "vad_positive_asr_empty"
    assert r["confidence"] == 0.0


def test_disagreement_flagged():
    r = compare_passes(
        {"p1": "the cost is fifty", "p2": "the cost is fifteen", "p3": "totally different words here"},
        {"p1": 0.8, "p2": 0.8, "p3": 0.8}, vad_positive=True, embedding_sim=0.2)
    assert r["disagreement"] is True
    assert r["flagged"] is True
    assert r["flag_reason"] == "cross_model_disagreement"


def test_low_confidence_flagged():
    r = compare_passes(
        {"p1": "same text", "p2": "same text", "p3": "same text"},
        {"p1": 0.2, "p2": 0.2, "p3": 0.2}, vad_positive=True, embedding_sim=0.95)
    assert r["flagged"] is True
    assert r["flag_reason"] == "low_logprob_confidence"
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `services/cross_model.py`**

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


def _max_pairwise_distance(texts):
    keys = list(texts)
    worst = 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = normalized_edit_distance(texts[keys[i]], texts[keys[j]])
            worst = max(worst, d)
    return worst


def compare_passes(texts, confidences, vad_positive, embedding_sim,
                   *, edit_thresh=0.4, sim_thresh=0.6, conf_thresh=0.5) -> dict:
    nonempty = {k: v for k, v in texts.items() if v and v.strip()}

    # 1. VAD said speech but a pass returned nothing -> lowest confidence.
    if vad_positive and len(nonempty) < len(texts):
        return {"confidence": 0.0, "disagreement": True, "flagged": True,
                "flag_reason": "vad_positive_asr_empty"}

    mean_conf = round(sum(confidences.values()) / max(len(confidences), 1), 3)
    worst = _max_pairwise_distance(nonempty) if len(nonempty) > 1 else 0.0
    disagreement = worst > edit_thresh and (embedding_sim is None or embedding_sim < sim_thresh)

    # 2. Cross-model disagreement.
    if disagreement:
        return {"confidence": mean_conf, "disagreement": True, "flagged": True,
                "flag_reason": "cross_model_disagreement"}
    # 3. Low confidence.
    if mean_conf < conf_thresh:
        return {"confidence": mean_conf, "disagreement": False, "flagged": True,
                "flag_reason": "low_logprob_confidence"}
    return {"confidence": mean_conf, "disagreement": False, "flagged": False,
            "flag_reason": None}
```

- [ ] **Step 4: Run, expect pass. Commit** `feat: phase4 cross-model compare + confidence (pure)`.

---

### Task 2: Overlap-aware speaker assignment (pure)

**Files:**
- Create: `services/diarize_assign.py`
- Test: `tests/test_diarize_assign.py`

**Interfaces:**
- `assign_speakers(segment: dict, turns: list[dict], min_overlap: float = 0.1) -> dict` returns `{"speakers": [...], "overlap": bool}` — every diarization turn overlapping the segment by ≥ `min_overlap` seconds contributes its speaker; `overlap=True` when ≥2 distinct speakers. Never empty: falls back to nearest turn's speaker as `["Unknown"]`-free single label.

- [ ] **Step 1: Failing test**

```python
# tests/test_diarize_assign.py
from services.diarize_assign import assign_speakers

TURNS = [
    {"start": 0.0, "end": 5.0, "speaker": "Speaker_1"},
    {"start": 4.0, "end": 9.0, "speaker": "Speaker_2"},
]


def test_single_speaker():
    r = assign_speakers({"start": 0.0, "end": 3.0}, TURNS)
    assert r == {"speakers": ["Speaker_1"], "overlap": False}


def test_overlap_two_speakers():
    r = assign_speakers({"start": 4.2, "end": 4.9}, TURNS)
    assert set(r["speakers"]) == {"Speaker_1", "Speaker_2"}
    assert r["overlap"] is True


def test_no_overlap_falls_back_to_nearest():
    r = assign_speakers({"start": 20.0, "end": 21.0}, TURNS)
    assert r["speakers"] == ["Speaker_2"]  # nearest by time
    assert r["overlap"] is False
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Implement `services/diarize_assign.py`**

```python
def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(segment, turns, min_overlap: float = 0.1) -> dict:
    s, e = segment["start"], segment["end"]
    hits = []
    for t in turns:
        ov = _overlap(s, e, t["start"], t["end"])
        if ov >= min_overlap:
            hits.append((ov, t["speaker"]))
    if hits:
        speakers, seen = [], set()
        for _, spk in sorted(hits, key=lambda x: -x[0]):
            if spk not in seen:
                seen.add(spk); speakers.append(spk)
        speakers.sort()
        return {"speakers": speakers, "overlap": len(speakers) > 1}

    # No overlap: nearest turn by time distance, never drop the segment.
    if not turns:
        return {"speakers": ["Speaker_1"], "overlap": False}
    nearest = min(turns, key=lambda t: min(abs(s - t["start"]), abs(e - t["end"])))
    return {"speakers": [nearest["speaker"]], "overlap": False}
```

- [ ] **Step 4: Run, expect pass. Commit** `feat: phase4 overlap-aware speaker assignment (pure)`.

---

### Task 3: Model wrappers — Indic ASR, embedding, diarization-overlap, clip cutter

**Files:**
- Create: `services/indic_asr_service.py`
- Create: `services/embedding_service.py`
- Create: `services/clip_service.py`
- Modify: `services/diarization_service.py` (add `diarize_with_overlap`)
- Modify: `config.py` (`INDIC_ASR_MODEL`, `EMBED_MODEL`)
- Test: `tests/test_clip_service.py`, `tests/test_phase4_wrappers_import.py`, `tests/test_config_phase4.py`

**Interfaces:**
- `clip_service.cut(wav_path, start, end, out_path) -> str` — ffmpeg slice (system tool, tested locally).
- `indic_asr_service.transcribe_clip(wav_path) -> {"text","confidence"}` (lazy).
- `embedding_service.similarity(a: str, b: str) -> float` (lazy LaBSE, cosine).
- `diarization_service.diarize_with_overlap(wav_path, num_speakers=None) -> list[{start,end,speaker}]` — pyannote 3.1 with overlapped-speech retained (multiple turns may cover one instant).
- `config.INDIC_ASR_MODEL`, `config.EMBED_MODEL`.

- [ ] **Step 1: Failing test — config**

```python
# tests/test_config_phase4.py
import config
def test_phase4_models():
    assert config.INDIC_ASR_MODEL
    assert config.EMBED_MODEL
```

- [ ] **Step 2: Failing test — clip cutter (ffmpeg allowed)**

```python
# tests/test_clip_service.py
import subprocess, shutil, wave
import pytest
from services.clip_service import cut
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")


def _tone(p):
    subprocess.run(["ffmpeg","-y","-f","lavfi","-i","sine=frequency=440:duration=3",
                    "-ar","16000", str(p)], capture_output=True, check=True)


def test_cut_extracts_subclip(tmp_path):
    src = tmp_path/"t.wav"; _tone(src)
    out = tmp_path/"clip.wav"
    cut(str(src), 0.5, 1.5, str(out))
    with wave.open(str(out)) as w:
        dur = w.getnframes()/w.getframerate()
    assert 0.8 < dur < 1.2
```

- [ ] **Step 3: Failing test — wrapper imports**

```python
# tests/test_phase4_wrappers_import.py
def test_imports():
    from services import indic_asr_service, embedding_service, diarization_service
    assert callable(indic_asr_service.transcribe_clip)
    assert callable(embedding_service.similarity)
    assert callable(diarization_service.diarize_with_overlap)
```

- [ ] **Step 4: Run all three, expect fail.**

- [ ] **Step 5: Append to `config.py`**

```python
# --- Phase 4: attribution + multi-pass ASR ---
INDIC_ASR_MODEL = os.getenv("INDIC_ASR_MODEL", "ai4bharat/indic-conformer-600m-multilingual")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/LaBSE")
```

- [ ] **Step 6: Implement `services/clip_service.py`**

```python
import subprocess


def cut(wav_path: str, start: float, end: float, out_path: str) -> str:
    dur = max(0.0, end - start)
    cmd = ["ffmpeg", "-y", "-i", wav_path, "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"clip cut failed: {r.stderr}")
    return out_path
```

- [ ] **Step 7: Implement `services/indic_asr_service.py`**

```python
import math
import config

_model = None


def load_indic():
    global _model
    if _model is None:
        from transformers import pipeline
        _model = pipeline("automatic-speech-recognition", model=config.INDIC_ASR_MODEL)
    return _model


def transcribe_clip(wav_path: str) -> dict:
    model = load_indic()
    out = model(wav_path)
    text = (out.get("text") if isinstance(out, dict) else str(out)) or ""
    # HF ASR pipeline gives no logprob; use a neutral mid confidence for the
    # third opinion (cross-model agreement, not this score, drives flagging).
    return {"text": text.strip(), "confidence": 0.5}
```

- [ ] **Step 8: Implement `services/embedding_service.py`**

```python
import config

_model = None


def load_embed():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    from sentence_transformers import util
    model = load_embed()
    emb = model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(emb[0], emb[1]).item())
```

- [ ] **Step 9: Append to `services/diarization_service.py`**

```python
def diarize_with_overlap(audio_path: str, num_speakers: int | None = None) -> list[dict]:
    """pyannote 3.1 turns WITH overlapped speech retained — overlapping instants
    yield multiple turns rather than being collapsed to one speaker."""
    pipeline = load_pipeline()
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    diarization = pipeline(audio_path, **kwargs)
    speaker_map, counter, segments = {}, 1, []
    for turn, _, label in diarization.itertracks(yield_label=True):
        if label not in speaker_map:
            speaker_map[label] = f"Speaker_{counter}"; counter += 1
        segments.append({"start": round(turn.start, 3), "end": round(turn.end, 3),
                         "speaker": speaker_map[label]})
    return segments
```

- [ ] **Step 10: Run, expect pass. Commit** `feat: phase4 model wrappers (indic/embed/overlap-diar) + clip cutter`.

---

### Task 4: Pipeline L4/L5/L6 integration + segment persistence

**Files:**
- Modify: `db/repository.py` (`add_segment`, `list_segments`)
- Modify: `pipeline/tasks.py` (L4/L5/L6; persist segments; write JSON artifacts; status NEEDS_REVIEW vs auto)
- Test: `tests/test_pipeline_attribution.py`

**Interfaces:**
- `repository.add_segment(session, *, file_id, start, end, speaker, text, confidence, source_pass, flagged, review_status) -> str`.
- `repository.list_segments(session, file_id) -> list[Segment]`.
- Pipeline after L3:
  - L4: `diarize_with_overlap(48k)`; write `diarization/{file_id}_speaker_timeline.json`.
  - L5: for each union segment, `clip_service.cut` from enhanced(or 16k) + original 16k; pass1 = whisper(enhanced clip), pass2 = whisper(original clip), pass3 = indic(original clip). (Whisper via existing `whisper_service.transcribe` on the clip; take first segment's text/confidence.)
  - L6: `compare_passes` (+ `embedding_service.similarity` between pass1/pass2); `assign_speakers`; persist `Segment` with winning text (pass1 if nonempty else pass2 else pass3), confidence, flag. Write `asr/passN_*` + `confidence/{file_id}_confidence_report.json`.
  - Reconcile union region count == segment count. Final status `NEEDS_REVIEW` if any flagged else `CERTIFIED`-eligible → set `NEEDS_REVIEW` only when flagged>0 else keep `NEEDS_REVIEW` (certification still requires Phase 5 sign-off; auto-accept only removes the flag).

- [ ] **Step 1: Failing test** (all models mocked via conftest + per-test)

```python
# tests/test_pipeline_attribution.py
import os, json, shutil, subprocess
import pytest
from db import base as dbbase, repository as repo
from db.models import JobStatus
from services import (audit_service as au, manifest_service as man, storage,
                      diarization_service, whisper_service, indic_asr_service,
                      embedding_service)
from pipeline import tasks as ptasks

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def setup_module():
    dbbase.init_db()


def _patch_store(monkeypatch, tmp_path):
    for mod in (ptasks.config, au.config, man.config, storage.config):
        monkeypatch.setattr(mod, "CASE_STORE_PATH", str(tmp_path))


def _tone(path):
    subprocess.run(["ffmpeg","-y","-f","lavfi","-i","sine=frequency=440:duration=2",
                    "-ar","22050", str(path)], capture_output=True, check=True)


def _mock_models(monkeypatch):
    monkeypatch.setattr(diarization_service, "diarize_with_overlap",
                        lambda p, num_speakers=None: [
                            {"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}])
    monkeypatch.setattr(whisper_service, "transcribe",
                        lambda path, **k: {"language": "te", "duration": 1.0,
                            "segments": [{"start": 0.0, "end": 1.0,
                            "text": "hello cost fifteen", "confidence": 0.9,
                            "no_speech_prob": 0.1, "compression_ratio": 1.2}]})
    monkeypatch.setattr(indic_asr_service, "transcribe_clip",
                        lambda p: {"text": "hello cost fifteen", "confidence": 0.5})
    monkeypatch.setattr(embedding_service, "similarity", lambda a, b: 0.95)


@pytest.mark.skipif(not HAS_FFMPEG, reason="no ffmpeg")
def test_segments_persisted_and_artifacts_written(tmp_path, monkeypatch):
    _patch_store(monkeypatch, tmp_path)
    _mock_models(monkeypatch)
    with dbbase.get_session() as s:
        case_id = repo.create_case(s)
        file_id = repo.create_file(s, case_id, "tone.wav", ".wav")
        job_id = repo.create_job(s, case_id, file_id)
        s.commit()
    inbox = os.path.join(str(tmp_path), "cases", case_id, "inbox")
    os.makedirs(inbox, exist_ok=True)
    _tone(os.path.join(inbox, f"{file_id}.wav"))

    assert ptasks.run_pipeline.apply(args=[job_id]).get() == JobStatus.NEEDS_REVIEW

    with dbbase.get_session() as s:
        segs = repo.list_segments(s, file_id)
        assert len(segs) == 1
        assert segs[0].speaker == "Speaker_1"
        assert segs[0].text == "hello cost fifteen"
        assert segs[0].flagged is False  # all passes agree, high conf
    conf = json.load(open(os.path.join(str(tmp_path), "cases", case_id,
        "derivatives", file_id, "confidence", f"{file_id}_confidence_report.json")))
    assert conf["segments_total"] == 1
    assert au.verify_chain(case_id) is True
```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Add repository helpers** — append to `db/repository.py`:

```python
from db.models import Segment


def add_segment(session, *, file_id, start, end, speaker, text, confidence,
                source_pass, flagged, review_status=None):
    seg = Segment(file_id=file_id, start=start, end=end, speaker=speaker,
                  text=text, confidence=confidence, source_pass=source_pass,
                  flagged=flagged, review_status=review_status)
    session.add(seg)
    session.flush()
    return seg.id


def list_segments(session, file_id):
    return (session.query(Segment).filter(Segment.file_id == file_id)
            .order_by(Segment.start).all())
```

- [ ] **Step 4: Edit `pipeline/tasks.py`** — add imports:

```python
from services import (diarization_service, whisper_service, indic_asr_service,
                      embedding_service, clip_service)
from services import cross_model, diarize_assign
```

- [ ] **Step 5: Add L4/L5/L6 to `pipeline/tasks.py`** (before `run_pipeline`)

```python
def _whisper_clip(clip_path, task):
    res = whisper_service.transcribe(clip_path, language="auto", use_vad=False, task=task)
    segs = res["segments"]
    if not segs:
        return {"text": "", "confidence": 0.0}
    return {"text": " ".join(s["text"] for s in segs).strip(),
            "confidence": round(sum(s["confidence"] for s in segs) / len(segs), 3)}


def _l4_diarize(job, in48, session):
    turns = diarization_service.diarize_with_overlap(in48)
    out = storage.derivative_path(job.case_id, job.file_id, "diarization",
                                  f"{job.file_id}_speaker_timeline.json")
    speakers = sorted({t["speaker"] for t in turns})
    with open(out, "w") as f:
        json.dump({"file_id": job.file_id, "speakers": speakers,
                   "timeline": turns,
                   "model_version": config.DIARIZATION_MODEL}, f, indent=2)
    au.append_entry(job.case_id, file_id=job.file_id, stage="L4",
                    model=config.DIARIZATION_MODEL,
                    parameters={"turns": len(turns)}, session=session)
    session.commit()
    return turns


def _l5_l6_segments(job, union, turns, enhanced16, original16, session):
    workdir = storage.derivative_path(job.case_id, job.file_id, "clips", "_")
    workdir = os.path.dirname(workdir)
    per_segment, flagged_count = [], 0
    enh_source = enhanced16 or original16
    for idx, region in enumerate(union):
        clip_enh = os.path.join(workdir, f"seg_{idx:04d}_enh.wav")
        clip_org = os.path.join(workdir, f"seg_{idx:04d}_org.wav")
        clip_service.cut(enh_source, region["start"], region["end"], clip_enh)
        clip_service.cut(original16, region["start"], region["end"], clip_org)

        p1 = _whisper_clip(clip_enh, "transcribe")
        p2 = _whisper_clip(clip_org, "transcribe")
        p3 = indic_asr_service.transcribe_clip(clip_org)
        texts = {"pass1_enhanced": p1["text"], "pass2_original": p2["text"],
                 "pass3_indic": p3["text"]}
        confs = {"pass1_enhanced": p1["confidence"], "pass2_original": p2["confidence"],
                 "pass3_indic": p3["confidence"]}
        sim = embedding_service.similarity(p1["text"], p2["text"])
        verdict = cross_model.compare_passes(texts, confs, vad_positive=True,
                                             embedding_sim=sim)
        spk = diarize_assign.assign_speakers(region, turns)
        winning = p1["text"] or p2["text"] or p3["text"]
        source_pass = ("pass1_enhanced" if p1["text"] else
                       "pass2_original" if p2["text"] else "pass3_indic")
        seg_id = repo.add_segment(
            session, file_id=job.file_id, start=region["start"], end=region["end"],
            speaker="+".join(spk["speakers"]), text=winning,
            confidence=verdict["confidence"], source_pass=source_pass,
            flagged=verdict["flagged"],
            review_status="pending" if verdict["flagged"] else "auto_accepted")
        if verdict["flagged"]:
            flagged_count += 1
        per_segment.append({
            "segment_id": seg_id, "edit_distance_norm": None,
            "embedding_similarity": round(sim, 3), "avg_logprob": None,
            "flag_reason": verdict["flag_reason"]})
    session.commit()
    reconcile.check("L3:union", len(union), "L5:segments", len(per_segment))
    return per_segment, flagged_count


def _write_confidence_report(job, per_segment, flagged_count, session):
    out = storage.derivative_path(job.case_id, job.file_id, "confidence",
                                  f"{job.file_id}_confidence_report.json")
    reasons = {}
    for ps in per_segment:
        if ps["flag_reason"]:
            reasons[ps["flag_reason"]] = reasons.get(ps["flag_reason"], 0) + 1
    with open(out, "w") as f:
        json.dump({"file_id": job.file_id, "segments_total": len(per_segment),
                   "segments_auto_accepted": len(per_segment) - flagged_count,
                   "segments_flagged": flagged_count, "flag_reasons": reasons,
                   "per_segment": per_segment}, f, indent=2)
    au.append_entry(job.case_id, file_id=job.file_id, stage="L6",
                    parameters={"flagged": flagged_count,
                                "total": len(per_segment)}, session=session)
    session.commit()
```

- [ ] **Step 6: Wire into `run_pipeline`** — replace the placeholder remainder with the L4/L5/L6 block (keep PLACEHOLDER_STAGES = ["L7","L8"]):

```python
    # L4/L5/L6 attribution + ASR + confidence.
    try:
        with get_session() as s:
            job = repo.get_job(s, job_id)
            in48 = storage.derivative_path(job.case_id, job.file_id, "normalized",
                                           f"{job.file_id}_48k.wav")
            in16 = storage.derivative_path(job.case_id, job.file_id, "normalized",
                                           f"{job.file_id}_16k_mono.wav")
            enh = storage.derivative_path(job.case_id, job.file_id, "enhanced",
                                          f"{job.file_id}_dfn3.wav")
            enh = enh if os.path.exists(enh) else None
            vad_json = storage.derivative_path(job.case_id, job.file_id, "vad",
                                               f"{job.file_id}_segments_union.json")
            union = json.load(open(vad_json))["segments"]

            repo.update_job(s, job_id, stage="L4"); s.commit()
            turns = _l4_diarize(job, in48, s)
            repo.update_job(s, job_id, stage="L5"); s.commit()
            per_segment, flagged = _l5_l6_segments(job, union, turns, enh, in16, s)
            repo.update_job(s, job_id, stage="L6"); s.commit()
            _write_confidence_report(job, per_segment, flagged, s)
    except Exception as e:
        with get_session() as s:
            repo.update_job(s, job_id, status=JobStatus.FAILED, error=str(e))
            s.commit()
        raise
```

- [ ] **Step 7: Run, expect pass** — `pytest tests/test_pipeline_attribution.py -v`.
- [ ] **Step 8: Full suite** — green except pre-existing alignment failure. (The conftest model stub must also stub the Phase 4 wrappers — see Step 9.)

- [ ] **Step 9: Extend conftest model stub** — in `tests/conftest.py` `_stub_models`, also stub Phase 4 wrappers so non-attribution pipeline tests stay model-free:

```python
    try:
        from services import (diarization_service, whisper_service,
                              indic_asr_service, embedding_service)
        monkeypatch.setattr(diarization_service, "diarize_with_overlap",
            lambda p, num_speakers=None: [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}],
            raising=False)
        monkeypatch.setattr(whisper_service, "transcribe",
            lambda path, **k: {"language": "te", "duration": 1.0, "segments": [
                {"start": 0.0, "end": 1.0, "text": "stub", "confidence": 0.9,
                 "no_speech_prob": 0.1, "compression_ratio": 1.0}]}, raising=False)
        monkeypatch.setattr(indic_asr_service, "transcribe_clip",
            lambda p: {"text": "stub", "confidence": 0.5}, raising=False)
        monkeypatch.setattr(embedding_service, "similarity", lambda a, b: 0.95, raising=False)
    except Exception:
        pass
```

- [ ] **Step 10: Commit** `feat: phase4 L4 diarize + L5 3-pass ASR + L6 confidence + segment persistence`.

---

## Phase 4 Done — Definition of Done

- Each VAD-union region → one persisted `Segment` (speaker(s), winning text, confidence, flag, review_status).
- Diarization timeline + confidence report JSON written; ledger covers L4/L5/L6.
- Flagging matches the three rules; agreement+high-confidence auto-accepts.
- Reconciliation: segment count == union region count.
- All prior tests green (pre-existing alignment failure excepted).

**Next:** Phase 5 (L8 certified-transcript output + L7 human QA REST + certification).
