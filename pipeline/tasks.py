import os
import json

import config
from pipeline.celery_app import celery
from pipeline import reconcile
from db.base import get_session
from db import repository as repo
from db.models import JobStatus
from services import audit_service as au
from services import storage
from services import manifest_service as man
from services import vad_service, enhancement_service, separation_service
from services import vad_union
from services import preprocess_service, hallucination_filter
from services import (diarization_service, indic_asr_service,
                      embedding_service, clip_service, lang_id_service)
from services import cross_model, whisper_service, asr_selector, telugu_asr_service
from services import transcript_service as ts
from services import webhook_service
from services.hashing import sha256_file
from services.ffmpeg_service import convert_dual_rate, UnsupportedFormatError


# Coarse progress percentage per stage — surfaced to webhooks / status polling so
# an integration can render a progress bar without knowing the stage internals.
STAGE_PROGRESS = {
    "L0": 5, "L1": 15, "L2": 30, "L2b": 35, "L3": 45,
    "L4": 55, "L5": 75, "L6": 85, "L8": 95,
    "completed": 100, "failed": 100, "quarantined": 100,
}


def _emit(callback_url, *, job_id, case_id, file_id, status, stage):
    """Best-effort job webhook. No-op when no callback_url was supplied."""
    if not callback_url:
        return
    webhook_service.notify(callback_url, {
        "job_id": job_id, "case_id": case_id, "file_id": file_id,
        "status": status, "stage": stage,
        "progress": STAGE_PROGRESS.get(stage if status == JobStatus.RUNNING else status, 0),
        "result_url": f"/jobs/{job_id}/result" if status == JobStatus.COMPLETED else None,
    })


def _inbox_original(case_id: str, file_id: str, ext: str) -> str:
    return os.path.join(config.CASE_STORE_PATH, "cases", case_id, "inbox",
                        f"{file_id}{ext}")


def _l0_ingest(job, session):
    """Hash the byte-exact original, WORM-store it, register manifest + file row.
    Returns (original_path, source_sha256). Raises FileNotFoundError if no
    staged input."""
    file_row = repo.get_file(session, job.file_id)
    ext = file_row.ext
    staged = _inbox_original(job.case_id, job.file_id, ext)
    if not os.path.exists(staged):
        raise FileNotFoundError(f"no staged original for file {job.file_id}")
    dest, digest = storage.write_original(job.case_id, job.file_id, ext, staged)
    man.register_file(job.case_id, job.file_id, file_row.original_filename, digest)
    repo.set_file_hash(session, job.file_id, digest)
    session.commit()
    au.append_entry(job.case_id, file_id=job.file_id, stage="L0",
                    output_hash=digest, session=session)
    session.commit()
    return dest, digest


def _l1_normalize(job, original_path: str, source_hash: str, session):
    out48 = storage.derivative_path(job.case_id, job.file_id, "normalized",
                                    f"{job.file_id}_48k.wav")
    out16 = storage.derivative_path(job.case_id, job.file_id, "normalized",
                                    f"{job.file_id}_16k_mono.wav")
    convert_dual_rate(original_path, out48, out16)
    for kind, path in [("normalized_48k", out48), ("normalized_16k", out16)]:
        h = sha256_file(path)
        man.register_derivative(job.case_id, job.file_id, kind, path, h,
                                parent_sha256=source_hash)
        au.append_entry(job.case_id, file_id=job.file_id, stage="L1",
                        input_hash=source_hash, output_hash=h, session=session)
    session.commit()
    reconcile.check("L0", 1, "L1", 1)
    return out48, out16


def _quarantine(job_id, case_id, file_id, reason: str):
    with get_session() as s:
        repo.update_job(s, job_id, status=JobStatus.QUARANTINED, error=reason)
        repo.set_file_status(s, file_id, "quarantined")
        s.commit()
    au.append_entry(case_id, file_id=file_id, stage="quarantine",
                    parameters={"reason": reason})


def _l2_enhance(job, in16, source_hash, session):
    """DeepFilterNet3 enhancement (parallel branch). On failure, flag degraded
    and return None so downstream runs original-only."""
    out = storage.derivative_path(job.case_id, job.file_id, "enhanced",
                                  f"{job.file_id}_dfn3.wav")
    try:
        enhancement_service.enhance(in16, out)
    except Exception as e:  # never fatal — original branch still carries recall
        repo.update_job(session, job.id, add_degraded="degraded_enhancement")
        session.commit()
        au.append_entry(job.case_id, file_id=job.file_id, stage="L2",
                        parameters={"error": str(e)}, session=session)
        session.commit()
        return None
    h = sha256_file(out)
    man.register_derivative(job.case_id, job.file_id, "enhanced_dfn3", out, h,
                            parent_sha256=source_hash)
    au.append_entry(job.case_id, file_id=job.file_id, stage="L2",
                    model=config.DFN_MODEL, input_hash=source_hash,
                    output_hash=h, session=session)
    session.commit()
    return out


def _l2b_separate(job, in16, source_hash, session):
    out = storage.derivative_path(job.case_id, job.file_id, "separated",
                                  f"{job.file_id}_vocal_stem.wav")
    try:
        separation_service.separate_vocals(in16, out)
    except Exception as e:
        au.append_entry(job.case_id, file_id=job.file_id, stage="L2b",
                        parameters={"error": str(e)}, session=session)
        session.commit()
        return None
    h = sha256_file(out)
    man.register_derivative(job.case_id, job.file_id, "separated_stem", out, h,
                            parent_sha256=source_hash)
    au.append_entry(job.case_id, file_id=job.file_id, stage="L2b",
                    model=config.DEMUCS_MODEL, input_hash=source_hash,
                    output_hash=h, session=session)
    session.commit()
    return out


def _l3_vad_union(job, in16, enhanced, stem, session):
    branches = {"original": vad_service.detect_speech(in16)}
    # Recall branches (enhanced / separated) are opt-in. The additive union adds
    # every region a branch calls speech; DeepFilterNet artifacts make Silero
    # over-detect, injecting phantom speech that ASR then hallucinates text onto.
    # For forensic evidence we anchor on the original track only by default.
    recall = config.VAD_INCLUDE_RECALL_BRANCHES
    if enhanced and recall:
        branches["enhanced"] = vad_service.detect_speech(enhanced)

    pre_union = vad_union.union_segments(list(branches.values()))
    separation_included = None
    if stem and recall:
        stem_segs = vad_service.detect_speech(stem)
        separation_included = vad_union.should_include_separation(
            len(pre_union), len(stem_segs))
        if separation_included:
            branches["separated"] = stem_segs

    union = vad_union.union_segments(list(branches.values()))
    branch_counts = {k: len(v) for k, v in branches.items()}
    # Additive guarantee: union covers at least as much duration as any branch.
    # (segment count may decrease when overlapping regions merge)
    for name, segs in branches.items():
        reconcile.check(f"L3:{name}", vad_union.total_duration(segs), "L3:union", vad_union.total_duration(union))

    out = storage.derivative_path(job.case_id, job.file_id, "vad",
                                  f"{job.file_id}_segments_union.json")
    payload = {"segments": union, "branch_counts": branch_counts,
               "separation_included": separation_included}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    au.append_entry(job.case_id, file_id=job.file_id, stage="L3",
                    parameters={"branch_counts": branch_counts,
                                "union_count": len(union)}, session=session)
    session.commit()
    return union


def _l4_diarize(job, in48, session):
    turns, model_version = diarization_service.diarize_with_overlap(in48)
    out = storage.derivative_path(job.case_id, job.file_id, "diarization",
                                  f"{job.file_id}_speaker_timeline.json")
    speakers = sorted({t["speaker"] for t in turns})
    with open(out, "w") as f:
        json.dump({"file_id": job.file_id, "speakers": speakers,
                   "timeline": turns,
                   "model_version": model_version}, f, indent=2)
    au.append_entry(job.case_id, file_id=job.file_id, stage="L4",
                    model=model_version,
                    parameters={"turns": len(turns)}, session=session)
    session.commit()
    return turns


def _partition_by_active_speakers(turns):
    """Partition the timeline into maximal intervals of constant active-speaker
    set, using a sweep over all turn boundaries. |set|==1 -> single speaker;
    |set|>=2 -> overlapped cross-talk. Adjacent equal-set intervals are merged."""
    if not turns:
        return []
    bounds = sorted({t["start"] for t in turns} | {t["end"] for t in turns})
    out = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a <= 1e-4:
            continue
        mid = (a + b) / 2.0
        active = sorted({t["speaker"] for t in turns
                         if t["start"] <= mid < t["end"]})
        if active:
            out.append({"start": a, "end": b, "speakers": active})
    merged = []
    for iv in out:
        if (merged and merged[-1]["speakers"] == iv["speakers"]
                and iv["start"] - merged[-1]["end"] <= 1e-3):
            merged[-1]["end"] = iv["end"]
        else:
            merged.append(dict(iv))
    return merged


def _subtract_interval(s, e, covered):
    """Sub-intervals of [s,e] not covered by the sorted (start,end) list."""
    result, cur = [], s
    for cs, ce in covered:
        if ce <= cur:
            continue
        if cs >= e:
            break
        if cs > cur:
            result.append((cur, min(cs, e)))
        cur = max(cur, ce)
        if cur >= e:
            break
    if cur < e:
        result.append((cur, e))
    return result


def _coalesce_speaker_units(units, same_speaker_gap_s):
    """Merge consecutive single-speaker units of the same speaker across small
    gaps, reducing fragmentation from the active-speaker partition."""
    out = []
    for u in units:
        if (out and u["type"] == "speaker" and out[-1]["type"] == "speaker"
                and out[-1]["speaker"] == u["speaker"]
                and u["start"] - out[-1]["end"] <= same_speaker_gap_s):
            out[-1]["end"] = u["end"]
        else:
            out.append(dict(u))
    return out


def _build_units(turns, vad_union, same_speaker_gap_s=None, min_dur_s=0.3):
    if same_speaker_gap_s is None:
        same_speaker_gap_s = config.DIARIZATION_SAME_SPEAKER_GAP_S
    """Typed transcription units covering ALL detected speech:
      - 'speaker': exactly one active speaker
      - 'overlap': two+ active speakers (cross-talk) -> SepFormer separation
      - 'gap'    : VAD speech with no diarization turn -> Speaker_unknown
    Gap recovery guarantees nothing inside the VAD union goes untranscribed."""
    units = []
    for iv in _partition_by_active_speakers(turns):
        if iv["end"] - iv["start"] < min_dur_s:
            continue
        units.append({
            "start": iv["start"], "end": iv["end"],
            "speaker": iv["speakers"][0], "speakers": iv["speakers"],
            "type": "overlap" if len(iv["speakers"]) > 1 else "speaker",
        })
    units = _coalesce_speaker_units(sorted(units, key=lambda u: u["start"]),
                                    same_speaker_gap_s)

    if config.GAP_RECOVERY_ENABLED:
        covered = sorted((u["start"], u["end"]) for u in units)
        for v in vad_union:
            for gs, ge in _subtract_interval(v["start"], v["end"], covered):
                if ge - gs < config.GAP_MIN_DUR_S:
                    continue
                t = gs
                while t < ge - 1e-6:
                    we = min(t + config.GAP_WINDOW_S, ge)
                    if we - t >= config.GAP_MIN_DUR_S:
                        units.append({
                            "start": round(t, 3), "end": round(we, 3),
                            "speaker": "Speaker_unknown",
                            "speakers": ["Speaker_unknown"], "type": "gap"})
                    t = we

    units.sort(key=lambda u: u["start"])
    return units


def run_asr(clean_clip, raw_clip, *, file_prior, whisper_turn=None, telugu_turn=None):
    """Dual-engine ASR — IndicConformer (per-clip) + Whisper-large-v3 (whole-file,
    sliced per turn) — with output-content selection.

    Language routing: trust clip MMS-LID above the gate (and within ALLOWED_LANGS
    when set), else the file prior. The routed language forces IndicConformer.

    IndicConformer runs twice (enhanced + original clip) for self-validation; the
    enhanced decode is primary, the original guards against over-suppression.

    Whisper is NOT run per clip (short-clip seq2seq decode degrades badly). The
    caller decodes the WHOLE file once with word timestamps and passes this turn's
    time-sliced text/confidence as `whisper_turn` (WhisperX pattern). asr_selector
    then picks: code-mix/numbers/entities -> Whisper, pure native Telugu ->
    IndicConformer (flagging genuine disagreement). `whisper_turn=None` (overlap
    stems, dual-engine off, or empty slice) -> legacy IndicConformer-only path.
    """
    mms = lang_id_service.identify(clean_clip)
    clip_lang = lang_id_service.to_iso639_1(mms.get("top1"))
    clip_conf = mms.get("top1_confidence") or 0.0

    # Trust per-clip LID only with enough confidence. Crucially, deviating from
    # the file prior (the language that dominates this recording) demands a HIGHER
    # bar than matching it — noise/short clips yield confident-but-wrong languages
    # that would otherwise poison the segment. Genuine code-switching still passes
    # at LID_DEVIATE_MIN_CONF. Generic: the prior is data-derived, no hardcoding.
    allowed = (not config.ALLOWED_LANGS or clip_lang in config.ALLOWED_LANGS)
    deviates = bool(clip_lang and file_prior and clip_lang != file_prior)
    needed_conf = config.LID_DEVIATE_MIN_CONF if deviates else config.LID_VOTE_MIN_CONF
    if clip_lang and allowed and clip_conf >= needed_conf:
        routing_lang = clip_lang
    else:
        routing_lang = file_prior  # may be None -> indic abstains

    i_enh = hallucination_filter.filter_pass(
        indic_asr_service.transcribe_clip(clean_clip, routing_lang))
    i_org = hallucination_filter.filter_pass(
        indic_asr_service.transcribe_clip(raw_clip, routing_lang))

    enh_text = (i_enh.get("text") or "").strip()
    org_text = (i_org.get("text") or "").strip()
    indic_text = enh_text or org_text          # enhanced primary, original fallback
    indic_source = "indic_enhanced" if enh_text else ("indic_original" if org_text else "none")
    indic_abstained = bool(i_enh.get("abstained") and i_org.get("abstained"))

    sc = cross_model.selfcheck_confidence(enh_text, org_text,
                                          embed_fn=embedding_service.similarity)
    hallucination = i_enh.get("hallucination") or i_org.get("hallucination")
    indic_model = i_enh.get("model") or i_org.get("model", config.INDIC_CONFORMER_MODEL)

    # Whisper output for this turn = the whole-file decode sliced by time (passed
    # in by the caller). Ghost-filter it so the same non-speech phrases blanked
    # for IndicConformer are also blanked here. None/empty -> indic-only path.
    whisper = None
    if config.ASR_DUAL_ENGINE and whisper_turn and (whisper_turn.get("text") or "").strip():
        wf = hallucination_filter.filter_pass({
            "text": whisper_turn.get("text", ""), "confidence": whisper_turn.get("confidence"),
            "language": routing_lang, "model": config.WHISPER_MODEL, "abstained": False})
        whisper = {"text": (wf.get("text") or "").strip(),
                   "confidence": whisper_turn.get("confidence"),
                   "no_speech_prob": None, "compression_ratio": None,
                   "language": routing_lang, "model": config.WHISPER_MODEL,
                   "hallucination": wf.get("hallucination")}
        if wf.get("hallucination"):
            hallucination = hallucination or wf.get("hallucination")

    # Third engine: fine-tuned Telugu Whisper turn-text (whole-file, sliced),
    # ghost-filtered the same way. Only present for ASR_FT_LANGS files.
    telugu = None
    if config.ASR_TELUGU_ENGINE and telugu_turn and (telugu_turn.get("text") or "").strip():
        tf = hallucination_filter.filter_pass({
            "text": telugu_turn.get("text", ""), "confidence": telugu_turn.get("confidence"),
            "language": routing_lang, "model": config.WHISPER_TELUGU_MODEL, "abstained": False})
        telugu = {"text": (tf.get("text") or "").strip(),
                  "confidence": telugu_turn.get("confidence"),
                  "language": routing_lang, "model": config.WHISPER_TELUGU_MODEL,
                  "hallucination": tf.get("hallucination")}
        if tf.get("hallucination"):
            hallucination = hallucination or tf.get("hallucination")

    has_external = (whisper and whisper["text"]) or (telugu and telugu["text"])
    if has_external:
        sel = asr_selector.select(
            indic_text=indic_text, indic_source=indic_source,
            indic_agreement=sc["agreement"], indic_abstained=indic_abstained,
            whisper_text=(whisper.get("text") if whisper else ""),
            whisper_confidence=(whisper.get("confidence") if whisper else None),
            whisper_no_speech=(whisper.get("no_speech_prob") if whisper else None),
            telugu_text=(telugu.get("text") if telugu else None),
            telugu_confidence=(telugu.get("confidence") if telugu else None),
            embed_fn=embedding_service.similarity,
            agreement_min=config.AGREEMENT_MIN, no_speech_max=config.NO_SPEECH_MAX)
        text, source = sel["text"], sel["source"]
        confidence = sel["confidence"] if sel["confidence"] is not None else sc["confidence"]
        agreement = sel["agreement"] if sel["agreement"] is not None else sc["agreement"]
        selection_reason, selection_flag = sel["reason"], sel["flag"]
        model = {"whisper": config.WHISPER_MODEL,
                 "telugu_whisper": config.WHISPER_TELUGU_MODEL}.get(source, indic_model)
        abstained = not (text or "").strip()
    else:
        text, source = indic_text, indic_source
        confidence, agreement = sc["confidence"], sc["agreement"]
        selection_reason, selection_flag = None, None
        model, abstained = indic_model, indic_abstained

    return {"lang_id": {"mms_top1": mms.get("top1"),
                        "mms_top1_confidence": mms.get("top1_confidence"),
                        "mms_top2": mms.get("top2"),
                        "mms_top2_confidence": mms.get("top2_confidence"),
                        "routing_lang": routing_lang},
            "text": text, "enh_text": enh_text, "org_text": org_text,
            "confidence": confidence, "agreement": agreement,
            "source": source, "abstained": abstained, "hallucination": hallucination,
            "model": model,
            # multi-engine extras (None on legacy path)
            "whisper": whisper, "telugu": telugu, "indic_text": indic_text,
            "indic_confidence": sc["confidence"], "indic_agreement": sc["agreement"],
            "selection_reason": selection_reason, "selection_flag": selection_flag}


# Backwards-compatible alias: the dual-engine path supersedes the indic-only one.
run_indic = run_asr


def _emit_segment(job, session, *, start, end, speaker, asr, clip_clean, clip_raw,
                  diarization_meta, extra_flags):
    """Persist one segment from the single-model IndicConformer result + self-check.
    Returns (segment_id, flagged, per_segment_entry)."""
    routing_lang = asr["lang_id"]["routing_lang"]

    # An external engine (generic Whisper or fine-tuned Telugu Whisper) won when
    # the chosen source is one of those — chosen on its own merits, so the
    # IndicConformer self-agreement suppression/divergence checks don't apply.
    external_won = asr["source"] in ("whisper", "telugu_whisper")

    # Forensic suppression: at/below the agreement floor the IndicConformer
    # enhanced and original passes share no signal -> the text is noise emitted
    # onto non-speech. Blank it (the clip is preserved for a reviewer) so the
    # record never carries fabricated words. Only applies when IndicConformer won.
    suppressed = False
    if (not external_won and asr["text"] and not asr["abstained"]
            and asr["agreement"] <= config.INDIC_SUPPRESS_BELOW):
        asr = {**asr, "text": "", "source": "suppressed_low_agreement"}
        suppressed = True

    candidates = {
        "lang_id": asr["lang_id"],
        # IndicConformer pass — records the engine's own enh/org outputs and
        # self-check (independent of which engine ultimately won the segment).
        "indic_conformer": {
            "text": asr.get("indic_text", asr["text"]),
            "enh_text": asr["enh_text"], "org_text": asr["org_text"],
            "confidence": asr.get("indic_confidence", asr["confidence"]),
            "agreement": asr.get("indic_agreement", asr["agreement"]),
            "language": routing_lang, "model": config.INDIC_CONFORMER_MODEL,
            "source": asr["source"], "abstained": asr["abstained"],
            "hallucination": asr["hallucination"]},
        "agreement": asr["agreement"],
        "selection_reason": asr.get("selection_reason"),
        "diarization": diarization_meta,
    }
    # Whisper pass (multi-engine only) — store its independent output + signals.
    if asr.get("whisper"):
        w = asr["whisper"]
        candidates["pass1_whisper"] = {
            "text": w.get("text", ""), "confidence": w.get("confidence"),
            "no_speech_prob": w.get("no_speech_prob"),
            "compression_ratio": w.get("compression_ratio"),
            "language": w.get("language"), "model": w.get("model")}
    # Fine-tuned Telugu Whisper pass.
    if asr.get("telugu"):
        t = asr["telugu"]
        candidates["pass_telugu_whisper"] = {
            "text": t.get("text", ""), "confidence": t.get("confidence"),
            "language": t.get("language"), "model": t.get("model")}

    reasons = []
    if asr["abstained"]:
        reasons.append("non_indic_abstain")
    elif suppressed:
        reasons.append("suppressed_low_agreement")
    elif not asr["text"]:
        reasons.append("asr_empty")
    else:
        # enh/org divergence is an IndicConformer self-check — only meaningful
        # when IndicConformer produced the winning text.
        if not external_won and asr["agreement"] < config.INDIC_SELFCHECK_MIN:
            reasons.append("enh_orig_divergence")
        if asr["confidence"] is not None and asr["confidence"] < config.INDIC_CONF_MIN:
            reasons.append("low_confidence")
    # Engine disagreement on pure-native text (selector flagged for review).
    if asr.get("selection_flag"):
        reasons.append(asr["selection_flag"])
    if asr["hallucination"]:
        reasons.append(asr["hallucination"])
    reasons.extend(extra_flags or [])
    flagged = bool(reasons)
    flag_reason = "+".join(dict.fromkeys(reasons)) if reasons else None

    _source_pass = {"whisper": "whisper", "telugu_whisper": "telugu_whisper"}.get(
        asr["source"], "indic_conformer")
    seg_id = repo.add_segment(
        session, file_id=job.file_id, start=start, end=end, speaker=speaker,
        text=asr["text"], confidence=asr["confidence"],
        source_pass=_source_pass, flagged=flagged,
        review_status="pending" if flagged else "auto_accepted",
        candidates=candidates, clip_original=clip_raw, clip_enhanced=clip_clean,
        detected_language=routing_lang)
    entry = {"segment_id": seg_id, "edit_distance_norm": None,
             "embedding_similarity": asr["agreement"], "avg_logprob": None,
             "flag_reason": flag_reason}
    return seg_id, flagged, entry


def _l5_l6_segments(job, union, turns, enhanced16, original16, session):
    workdir = os.path.dirname(
        storage.derivative_path(job.case_id, job.file_id, "clips", "_"))
    per_segment, flagged_count = [], 0
    enh_source = enhanced16 or original16

    # Hybrid segmentation: active-speaker partition (single + overlap units)
    # plus gap recovery for VAD speech pyannote left unattributed.
    units = _build_units(turns, union)

    import torch as _torch

    # Robust preprocessing per unit (denoise + loudnorm + edge-trim) -> one clean
    # clip fed to all 3 models. Then a cheap MMS-LID pre-sweep over every clean
    # clip yields a file-level language prior that tames per-clip LID misroutes.
    prepared = []
    lids = []
    for idx, unit in enumerate(units):
        clips = preprocess_service.prepare_clip(
            enh_source, original16, unit["start"], unit["end"], workdir, idx, unit["speaker"])
        prepared.append(clips)
        lids.append(lang_id_service.identify(clips["clean"]))
    file_prior = lang_id_service.vote_file_language(
        lids, allowed_langs=config.ALLOWED_LANGS, min_conf=config.LID_VOTE_MIN_CONF)

    # WhisperX-pattern second engine: decode the WHOLE file once with word
    # timestamps (full context -> fluent, punctuated, correct numbers), on the
    # RAW 16k track (speech enhancement degrades Whisper). Each turn's text is
    # then sliced from these words by time and handed to run_asr. One decode for
    # the file beats short per-clip decodes that lose all context.
    whisper_words = []
    if config.ASR_DUAL_ENGINE and file_prior:
        whisper_words = whisper_service.transcribe_words(original16, file_prior).get("words", [])

    # Third engine: fine-tuned Telugu Whisper, whole-file word timestamps, on the
    # RAW track. Only for files whose language prior is a fine-tune target (the
    # fine-tune is Telugu-only). Sliced per turn like the generic Whisper pass.
    telugu_words = []
    if config.ASR_TELUGU_ENGINE and file_prior in config.ASR_FT_LANGS:
        telugu_words = telugu_asr_service.transcribe_words(original16, file_prior).get("words", [])

    for idx, unit in enumerate(units):
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        utype = unit["type"]
        speaker = unit["speaker"]
        clips = prepared[idx]

        # Cross-talk: split into per-speaker streams, transcribe each one so
        # every overlapped voice is recovered (not just the loudest).
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
                    asr = run_asr(stem, stem, file_prior=file_prior)
                    meta = {"speaker": spk,
                            "concurrent_speakers": [s for s in unit["speakers"] if s != spk],
                            "is_overlap": True, "segment_type": "overlap",
                            "separation": "sepformer", "stem_index": si}
                    _, flagged, entry = _emit_segment(
                        job, session, start=unit["start"], end=unit["end"],
                        speaker=spk, asr=asr, clip_clean=stem, clip_raw=stem,
                        diarization_meta=meta, extra_flags=["overlapping_speech"])
                    flagged_count += 1 if flagged else 0
                    per_segment.append(entry)
                continue
            # Separation failed -> fall through to mixed-clip transcription.

        # Slice this turn's text from each whole-file word stream by time.
        whisper_turn = (whisper_service.slice_words(whisper_words, unit["start"], unit["end"])
                        if whisper_words else None)
        telugu_turn = (whisper_service.slice_words(telugu_words, unit["start"], unit["end"])
                       if telugu_words else None)
        asr = run_asr(clips["clean"], clips["raw"], file_prior=file_prior,
                      whisper_turn=whisper_turn, telugu_turn=telugu_turn)
        extra = []
        if utype == "overlap":
            extra.append("overlapping_speech")
        if utype == "gap":
            extra.append("gap_recovery")
        meta = {"speaker": speaker,
                "concurrent_speakers": [s for s in unit["speakers"] if s != speaker],
                "is_overlap": utype == "overlap", "segment_type": utype}
        _, flagged, entry = _emit_segment(
            job, session, start=unit["start"], end=unit["end"],
            speaker=speaker, asr=asr, clip_clean=clips["clean"], clip_raw=clips["raw"],
            diarization_meta=meta, extra_flags=extra)
        flagged_count += 1 if flagged else 0
        per_segment.append(entry)

    session.commit()
    reconcile.check("L4:units", len(units), "L5:segments", len(per_segment))
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


@celery.task(name="pipeline.run_pipeline")
def run_pipeline(job_id: str) -> str:
    with get_session() as s:
        job = repo.get_job(s, job_id)
        if job is None:
            raise ValueError(f"job not found: {job_id}")
        case_id, file_id = job.case_id, job.file_id
        callback_url = (job.options or {}).get("callback_url")
        repo.update_job(s, job_id, status=JobStatus.RUNNING, stage="L0")
        s.commit()
    _emit(callback_url, job_id=job_id, case_id=case_id, file_id=file_id,
          status=JobStatus.RUNNING, stage="L0")

    def stage(st):
        _emit(callback_url, job_id=job_id, case_id=case_id, file_id=file_id,
              status=JobStatus.RUNNING, stage=st)

    # L0 + L1 with quarantine on bad/missing input.
    try:
        with get_session() as s:
            job = repo.get_job(s, job_id)
            original_path, source_hash = _l0_ingest(job, s)
        with get_session() as s:
            job = repo.get_job(s, job_id)
            repo.update_job(s, job_id, stage="L1")
            s.commit()
            _l1_normalize(job, original_path, source_hash, s)
        stage("L1")
    except (FileNotFoundError, UnsupportedFormatError, RuntimeError) as e:
        _quarantine(job_id, case_id, file_id, str(e))
        _emit(callback_url, job_id=job_id, case_id=case_id, file_id=file_id,
              status=JobStatus.QUARANTINED, stage="quarantine")
        return JobStatus.QUARANTINED

    # L2/L2b/L3 recall branches.
    try:
        with get_session() as s:
            job = repo.get_job(s, job_id)
            in16 = storage.derivative_path(job.case_id, job.file_id, "normalized",
                                           f"{job.file_id}_16k_mono.wav")
            in48 = storage.derivative_path(job.case_id, job.file_id, "normalized",
                                           f"{job.file_id}_48k.wav")
            repo.update_job(s, job_id, stage="L2"); s.commit(); stage("L2")
            enhanced = _l2_enhance(job, in16, source_hash, s)
            stem = None
            if (job.options or {}).get("separate"):
                repo.update_job(s, job_id, stage="L2b"); s.commit(); stage("L2b")
                stem = _l2b_separate(job, in48, source_hash, s)
            repo.update_job(s, job_id, stage="L3"); s.commit(); stage("L3")
            _l3_vad_union(job, in16, enhanced, stem, s)
    except Exception as e:
        with get_session() as s:
            repo.update_job(s, job_id, status=JobStatus.FAILED, error=str(e))
            s.commit()
        _emit(callback_url, job_id=job_id, case_id=case_id, file_id=file_id,
              status=JobStatus.FAILED, stage="L3")
        raise

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

            repo.update_job(s, job_id, stage="L4"); s.commit(); stage("L4")
            turns = _l4_diarize(job, in48, s)
            repo.update_job(s, job_id, stage="L5"); s.commit(); stage("L5")
            per_segment, flagged = _l5_l6_segments(job, union, turns, enh, in16, s)
            repo.update_job(s, job_id, stage="L6"); s.commit(); stage("L6")
            _write_confidence_report(job, per_segment, flagged, s)
    except Exception as e:
        with get_session() as s:
            repo.update_job(s, job_id, status=JobStatus.FAILED, error=str(e))
            s.commit()
        _emit(callback_url, job_id=job_id, case_id=case_id, file_id=file_id,
              status=JobStatus.FAILED, stage="L5")
        raise

    # L8 output generation.
    try:
        with get_session() as s:
            job = repo.get_job(s, job_id)
            repo.update_job(s, job_id, stage="L8"); s.commit(); stage("L8")
            segs = repo.list_segments(s, job.file_id)
            src_hash = repo.get_file(s, job.file_id).source_sha256
            data = ts.build(job.case_id, job.file_id, src_hash, segs,
                            status="completed")
            ts.write(job.case_id, job.file_id, data)            # certified_transcript.json
            # IndicConformer transcript + single-model self-check validation report.
            ts.write_named(job.case_id, job.file_id, "indic_transcript", data)
            ts.write_named(job.case_id, job.file_id, "validation_report",
                           ts.build_indic_validation(job.file_id, segs))
            # Court-ready Time/Person/Conversation table (JSON + Markdown).
            table = ts.build_conversation_table(job.file_id, segs)
            ts.write_named(job.case_id, job.file_id, "conversation_table", table)
            ts.write_named_text(job.case_id, job.file_id, "conversation_table", "md",
                                ts.render_conversation_markdown(table))
            au.append_entry(job.case_id, file_id=job.file_id, stage="L8",
                            parameters={"segments": len(segs)}, session=s)
            s.commit()
            # No human-review gate: the pipeline completes and results are ready.
            repo.update_job(s, job_id, status=JobStatus.COMPLETED); s.commit()
        _emit(callback_url, job_id=job_id, case_id=case_id, file_id=file_id,
              status=JobStatus.COMPLETED, stage="L8")
        return JobStatus.COMPLETED
    except Exception as e:
        with get_session() as s:
            repo.update_job(s, job_id, status=JobStatus.FAILED, error=str(e)); s.commit()
        _emit(callback_url, job_id=job_id, case_id=case_id, file_id=file_id,
              status=JobStatus.FAILED, stage="L8")
        raise
