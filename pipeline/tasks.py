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


def _build_turn_segments(turns, vad_union,
                          same_speaker_gap_s=0.5,
                          min_dur_s=0.3,
                          vad_min_overlap_s=0.1):
    """
    Convert diarization turns into transcription units.

    - Merge adjacent same-speaker turns separated by <= same_speaker_gap_s.
    - Drop turns shorter than min_dur_s.
    - Validate each turn has VAD-confirmed speech (vad_min_overlap_s).
    - Overlapping turns from DIFFERENT speakers are preserved separately so
      both voices get independently transcribed (each model will pick up the
      dominant voice in that window — true source-separated transcription
      requires a future per-turn Demucs pass).
    """
    if not turns:
        return []

    sorted_turns = sorted(turns, key=lambda t: t["start"])
    merged = [dict(sorted_turns[0])]
    for t in sorted_turns[1:]:
        last = merged[-1]
        gap = t["start"] - last["end"]
        if (t["speaker"] == last["speaker"]
                and 0 <= gap <= same_speaker_gap_s):
            last["end"] = max(last["end"], t["end"])
        else:
            merged.append(dict(t))

    result = []
    for m in merged:
        if m["end"] - m["start"] < min_dur_s:
            continue
        has_vad = any(
            max(0.0, min(m["end"], v["end"]) - max(m["start"], v["start"]))
            >= vad_min_overlap_s
            for v in vad_union
        )
        if has_vad:
            result.append(m)
    return result


def _overlapping_speakers(unit, all_units):
    """Return speakers whose turns overlap with this unit (excluding self)."""
    s, e = unit["start"], unit["end"]
    others = set()
    for u in all_units:
        if u is unit:
            continue
        if u["speaker"] != unit["speaker"] and u["start"] < e and u["end"] > s:
            others.add(u["speaker"])
    return sorted(others)


def _l5_l6_segments(job, union, turns, enhanced16, original16, session):
    workdir = os.path.dirname(
        storage.derivative_path(job.case_id, job.file_id, "clips", "_"))
    per_segment, flagged_count = [], 0
    enh_source = enhanced16 or original16

    # Use diarization turns (not VAD blobs) as segmentation units.
    # Each turn = one clip = one speaker = one transcription row.
    # Overlapping turns from different speakers generate separate clips,
    # so both voices are independently transcribed.
    seg_units = _build_turn_segments(turns, union)

    for idx, unit in enumerate(seg_units):
        speaker = unit["speaker"]
        concurrent = _overlapping_speakers(unit, seg_units)

        clip_enh = os.path.join(workdir, f"seg_{idx:04d}_{speaker}_enh.wav")
        clip_org = os.path.join(workdir, f"seg_{idx:04d}_{speaker}_org.wav")
        clip_service.cut(enh_source, unit["start"], unit["end"], clip_enh)
        clip_service.cut(original16, unit["start"], unit["end"], clip_org)

        # Independent language ID via MMS-LID (audio-grounded, not decoder-dependent).
        mms = lang_id_service.identify(clip_enh)
        mms_top1 = mms.get("top1")
        routing_lang_2 = lang_id_service.to_iso639_1(mms_top1)

        # Pass 1: Whisper large-v3, enhanced audio, auto language detect.
        p1 = _clean_pass(_whisper_clip(clip_enh, "transcribe"))
        whisper_lang = p1.get("language") or "und"

        routing_lang = routing_lang_2 or whisper_lang
        lang_mismatch = bool(
            routing_lang_2 and whisper_lang != "und"
            and routing_lang_2 != whisper_lang
        )

        # Pass 2: IndicConformer-600M, language-routed by MMS-LID.
        p2 = _clean_pass(indic_asr_service.transcribe_clip(clip_enh, routing_lang))

        # Pass 3: SeamlessM4T v2 on ORIGINAL audio, language-routed by MMS-LID.
        p3 = _clean_pass(seamless_service.transcribe_clip(clip_org, routing_lang))

        detected_lang = routing_lang

        candidates = {
            "lang_id": {
                "mms_top1": mms_top1,
                "mms_top1_confidence": mms.get("top1_confidence"),
                "mms_top2": mms.get("top2"),
                "mms_top2_confidence": mms.get("top2_confidence"),
                "whisper_lang": whisper_lang,
                "routing_lang": routing_lang,
                "lang_mismatch": lang_mismatch,
            },
            "pass1_whisper": {
                "text": p1["text"], "confidence": p1["confidence"], "language": whisper_lang,
                "model": "openai/whisper-large-v3",
            },
            "pass2_indic_conformer": {
                "text": p2["text"], "confidence": p2["confidence"], "language": routing_lang,
                "model": p2.get("model", config.INDIC_CONFORMER_MODEL),
            },
            "pass3_seamless": {
                "text": p3["text"], "confidence": p3["confidence"], "language": routing_lang,
                "model": config.SEAMLESS_MODEL, "audio_source": "original",
            },
            "diarization": {
                "speaker": speaker,
                "concurrent_speakers": concurrent,
                "is_overlap": bool(concurrent),
            },
        }
        texts = {
            "pass1_whisper": p1["text"],
            "pass2_indic_conformer": p2["text"],
            "pass3_seamless": p3["text"],
        }
        confs = {
            "pass1_whisper": p1["confidence"],
            "pass2_indic_conformer": p2["confidence"],
            "pass3_seamless": p3["confidence"],
        }

        sim = embedding_service.similarity(p1["text"], p2["text"])
        verdict = cross_model.compare_passes(texts, confs, vad_positive=True,
                                             embedding_sim=sim)

        if lang_mismatch and not verdict["flagged"]:
            verdict = {**verdict, "flagged": True, "flag_reason": "lang_id_mismatch"}
        elif lang_mismatch and verdict["flagged"]:
            verdict = {**verdict, "flag_reason": verdict["flag_reason"] + "+lang_id_mismatch"}

        # Flag overlapping-speech segments for human review.
        if concurrent and not verdict["flagged"]:
            verdict = {**verdict, "flagged": True, "flag_reason": "overlapping_speech"}
        elif concurrent and verdict["flagged"]:
            verdict = {**verdict, "flag_reason": verdict["flag_reason"] + "+overlapping_speech"}

        winning = p1["text"] or p2["text"] or p3["text"]
        source_pass = "pass1_whisper"

        seg_id = repo.add_segment(
            session, file_id=job.file_id, start=unit["start"], end=unit["end"],
            speaker=speaker, text=winning,
            confidence=verdict["confidence"], source_pass=source_pass,
            flagged=verdict["flagged"],
            review_status="pending" if verdict["flagged"] else "auto_accepted",
            candidates=candidates, clip_original=clip_org, clip_enhanced=clip_enh,
            detected_language=detected_lang)
        if verdict["flagged"]:
            flagged_count += 1
        per_segment.append({
            "segment_id": seg_id, "edit_distance_norm": None,
            "embedding_similarity": round(sim, 3), "avg_logprob": None,
            "flag_reason": verdict["flag_reason"]})

    session.commit()
    # Sanity check: at least one segment produced
    reconcile.check("L4:turns", len(seg_units), "L5:segments", len(per_segment))
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
