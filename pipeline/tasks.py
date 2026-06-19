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
from services import (diarization_service, whisper_service, indic_asr_service,
                      seamless_service, embedding_service, clip_service,
                      lang_id_service)
from services import cross_model
from services import transcript_service as ts
from services.hashing import sha256_file
from services.ffmpeg_service import convert_dual_rate, UnsupportedFormatError


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
    if enhanced:
        branches["enhanced"] = vad_service.detect_speech(enhanced)

    pre_union = vad_union.union_segments(list(branches.values()))
    separation_included = None
    if stem:
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


def _whisper_clip(clip_path, task):
    res = whisper_service.transcribe(clip_path, language="auto", use_vad=False, task=task)
    segs = res["segments"]
    if not segs:
        return {"text": "", "confidence": 0.0, "language": res.get("language", "und")}
    return {"text": " ".join(s["text"] for s in segs).strip(),
            "confidence": round(sum(s["confidence"] for s in segs) / len(segs), 3),
            "language": res.get("language", "und")}


def _has_repetition(text: str) -> bool:
    """Detect degenerate hallucination loops: consecutive repeats or extreme lexical monotony."""
    if not text:
        return False
    words = text.split()
    if len(words) < 4:
        return False
    # 3+ consecutive identical tokens
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return True
    # Unique ratio <30% with enough words → hallucination loop
    if len(words) >= 8 and len(set(words)) / len(words) < 0.30:
        return True
    return False


def _clean_pass(result: dict) -> dict:
    """Blank pass output if repetition loop detected; zero confidence."""
    if _has_repetition(result.get("text", "")):
        return {
            **result,
            "text": "",
            "confidence": 0.0,
            "repetition_detected": True,
        }
    return result


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


def _build_units(turns, vad_union, same_speaker_gap_s=0.5, min_dur_s=0.3):
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


def _three_pass_asr(clip_enh, clip_org):
    """MMS-LID + 3 independent ASR passes on a prepared clip pair. clip_org may
    equal clip_enh (separated-speaker streams have no distinct original source)."""
    mms = lang_id_service.identify(clip_enh)
    mms_top1 = mms.get("top1")
    mms_top1_conf = mms.get("top1_confidence") or 0.0
    routing_lang_2 = lang_id_service.to_iso639_1(mms_top1)

    p1 = _clean_pass(_whisper_clip(clip_enh, "transcribe"))
    whisper_lang = p1.get("language") or "und"

    # Trust MMS-LID routing only above the confidence floor; else Whisper detect.
    if routing_lang_2 and mms_top1_conf >= config.MMS_LID_MIN_CONFIDENCE:
        routing_lang = routing_lang_2
    else:
        routing_lang = whisper_lang or "und"

    lang_mismatch = bool(
        routing_lang_2 and mms_top1_conf >= config.MMS_LID_MIN_CONFIDENCE
        and whisper_lang != "und" and routing_lang_2 != whisper_lang)

    p2 = _clean_pass(indic_asr_service.transcribe_clip(clip_enh, routing_lang))
    p3 = _clean_pass(seamless_service.transcribe_clip(clip_org, routing_lang))

    return {"mms": mms, "mms_top1": mms_top1, "whisper_lang": whisper_lang,
            "routing_lang": routing_lang, "lang_mismatch": lang_mismatch,
            "p1": p1, "p2": p2, "p3": p3}


def _emit_segment(job, session, *, start, end, speaker, asr, clip_enh, clip_org,
                  pass3_source, diarization_meta, extra_flags):
    """Build candidates + cross-model verdict and persist one segment row.
    Returns (segment_id, flagged, per_segment_entry)."""
    p1, p2, p3 = asr["p1"], asr["p2"], asr["p3"]
    whisper_lang, routing_lang, mms = asr["whisper_lang"], asr["routing_lang"], asr["mms"]

    candidates = {
        "lang_id": {
            "mms_top1": asr["mms_top1"],
            "mms_top1_confidence": mms.get("top1_confidence"),
            "mms_top2": mms.get("top2"),
            "mms_top2_confidence": mms.get("top2_confidence"),
            "whisper_lang": whisper_lang,
            "routing_lang": routing_lang,
            "lang_mismatch": asr["lang_mismatch"],
        },
        "pass1_whisper": {"text": p1["text"], "confidence": p1["confidence"],
                          "language": whisper_lang, "model": "openai/whisper-large-v3"},
        "pass2_indic_conformer": {"text": p2["text"], "confidence": p2["confidence"],
                                  "language": routing_lang,
                                  "model": p2.get("model", config.INDIC_CONFORMER_MODEL)},
        "pass3_seamless": {"text": p3["text"], "confidence": p3["confidence"],
                           "language": routing_lang, "model": config.SEAMLESS_MODEL,
                           "audio_source": pass3_source},
        "diarization": diarization_meta,
    }
    texts = {"pass1_whisper": p1["text"], "pass2_indic_conformer": p2["text"],
             "pass3_seamless": p3["text"]}
    confs = {"pass1_whisper": p1["confidence"], "pass2_indic_conformer": p2["confidence"],
             "pass3_seamless": p3["confidence"]}
    sim = embedding_service.similarity(p1["text"], p2["text"])
    verdict = cross_model.compare_passes(texts, confs, vad_positive=True,
                                         embedding_sim=sim)

    reasons = []
    if verdict["flagged"] and verdict.get("flag_reason"):
        reasons.append(verdict["flag_reason"])
    if asr["lang_mismatch"]:
        reasons.append("lang_id_mismatch")
    reasons.extend(extra_flags or [])
    flagged = bool(reasons)
    flag_reason = "+".join(dict.fromkeys(reasons)) if reasons else verdict.get("flag_reason")

    winning = p1["text"] or p2["text"] or p3["text"]
    seg_id = repo.add_segment(
        session, file_id=job.file_id, start=start, end=end,
        speaker=speaker, text=winning,
        confidence=verdict["confidence"], source_pass="pass1_whisper",
        flagged=flagged,
        review_status="pending" if flagged else "auto_accepted",
        candidates=candidates, clip_original=clip_org, clip_enhanced=clip_enh,
        detected_language=routing_lang)
    entry = {"segment_id": seg_id, "edit_distance_norm": None,
             "embedding_similarity": round(sim, 3), "avg_logprob": None,
             "flag_reason": flag_reason}
    return seg_id, flagged, entry


def _l5_l6_segments(job, union, turns, enhanced16, original16, session):
    workdir = os.path.dirname(
        storage.derivative_path(job.case_id, job.file_id, "clips", "_"))
    per_segment, flagged_count = [], 0
    enh_source = enhanced16 or original16
    norm = config.CLIP_NORMALIZE

    # Hybrid segmentation: active-speaker partition (single + overlap units)
    # plus gap recovery for VAD speech pyannote left unattributed.
    units = _build_units(turns, union)

    import torch as _torch

    for idx, unit in enumerate(units):
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        utype = unit["type"]
        speaker = unit["speaker"]
        clip_enh = os.path.join(workdir, f"seg_{idx:04d}_{speaker}_enh.wav")
        clip_org = os.path.join(workdir, f"seg_{idx:04d}_{speaker}_org.wav")
        clip_service.cut(enh_source, unit["start"], unit["end"], clip_enh, normalize=norm)
        clip_service.cut(original16, unit["start"], unit["end"], clip_org, normalize=norm)

        # Cross-talk: split into per-speaker streams, transcribe each one so
        # every overlapped voice is recovered (not just the loudest).
        if (utype == "overlap" and config.OVERLAP_SEPARATION_ENABLED
                and (unit["end"] - unit["start"]) >= config.OVERLAP_MIN_DUR_S):
            stems = separation_service.separate_speakers(
                clip_enh, workdir, f"seg_{idx:04d}")
            if stems:
                for si, stem in enumerate(stems):
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                    spk = (unit["speakers"][si] if si < len(unit["speakers"])
                           else f"overlap_spk{si}")
                    asr = _three_pass_asr(stem, stem)
                    meta = {"speaker": spk,
                            "concurrent_speakers": [s for s in unit["speakers"] if s != spk],
                            "is_overlap": True, "segment_type": "overlap",
                            "separation": "sepformer", "stem_index": si}
                    _, flagged, entry = _emit_segment(
                        job, session, start=unit["start"], end=unit["end"],
                        speaker=spk, asr=asr, clip_enh=stem, clip_org=stem,
                        pass3_source="separated_stem", diarization_meta=meta,
                        extra_flags=["overlapping_speech"])
                    flagged_count += 1 if flagged else 0
                    per_segment.append(entry)
                continue
            # Separation failed -> fall through to mixed-clip transcription.

        asr = _three_pass_asr(clip_enh, clip_org)
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
            speaker=speaker, asr=asr, clip_enh=clip_enh, clip_org=clip_org,
            pass3_source="original", diarization_meta=meta, extra_flags=extra)
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
        repo.update_job(s, job_id, status=JobStatus.RUNNING, stage="L0")
        s.commit()

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
    except (FileNotFoundError, UnsupportedFormatError, RuntimeError) as e:
        _quarantine(job_id, case_id, file_id, str(e))
        return JobStatus.QUARANTINED

    # L2/L2b/L3 recall branches.
    try:
        with get_session() as s:
            job = repo.get_job(s, job_id)
            in16 = storage.derivative_path(job.case_id, job.file_id, "normalized",
                                           f"{job.file_id}_16k_mono.wav")
            in48 = storage.derivative_path(job.case_id, job.file_id, "normalized",
                                           f"{job.file_id}_48k.wav")
            repo.update_job(s, job_id, stage="L2"); s.commit()
            enhanced = _l2_enhance(job, in16, source_hash, s)
            stem = None
            if (job.options or {}).get("separate"):
                repo.update_job(s, job_id, stage="L2b"); s.commit()
                stem = _l2b_separate(job, in48, source_hash, s)
            repo.update_job(s, job_id, stage="L3"); s.commit()
            _l3_vad_union(job, in16, enhanced, stem, s)
    except Exception as e:
        with get_session() as s:
            repo.update_job(s, job_id, status=JobStatus.FAILED, error=str(e))
            s.commit()
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

    # L8 output generation.
    try:
        with get_session() as s:
            job = repo.get_job(s, job_id)
            repo.update_job(s, job_id, stage="L8"); s.commit()
            segs = repo.list_segments(s, job.file_id)
            src_hash = repo.get_file(s, job.file_id).source_sha256
            data = ts.build(job.case_id, job.file_id, src_hash, segs,
                            status="machine_assisted_pending_certification")
            ts.write(job.case_id, job.file_id, data)
            au.append_entry(job.case_id, file_id=job.file_id, stage="L8",
                            parameters={"segments": len(segs)}, session=s)
            s.commit()
            repo.update_job(s, job_id, status=JobStatus.NEEDS_REVIEW); s.commit()
        return JobStatus.NEEDS_REVIEW
    except Exception as e:
        with get_session() as s:
            repo.update_job(s, job_id, status=JobStatus.FAILED, error=str(e)); s.commit()
        raise
