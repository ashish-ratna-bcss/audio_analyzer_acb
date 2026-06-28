import os
import json as _json
import shutil

import aiofiles
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, status

import config
from api.auth import require_api_key
from db.base import get_session
from db import repository as repo
from db.models import Case, JobStatus
from services import audit_service as au
from services import storage
from services import transcript_service as ts
from pipeline.tasks import run_pipeline, STAGE_PROGRESS

router = APIRouter()

# Terminal job states — the integration can stop polling once a job reaches one.
_TERMINAL = {JobStatus.COMPLETED, JobStatus.CERTIFIED, JobStatus.FAILED,
             JobStatus.QUARANTINED}


def _job_view(job) -> dict:
    """Status payload shared by the status endpoint and the webhook."""
    done = job.status in _TERMINAL
    progress = STAGE_PROGRESS.get(
        job.stage if job.status == JobStatus.RUNNING else job.status, 0)
    return {
        "job_id": job.id,
        "case_id": job.case_id,
        "file_id": job.file_id,
        "status": job.status,
        "stage": job.stage,
        "progress": progress,
        "degraded_flags": job.degraded_flags or [],
        "error": job.error,
        "is_terminal": done,
        "result_url": f"/jobs/{job.id}/result" if job.status in (
            JobStatus.COMPLETED, JobStatus.CERTIFIED) else None,
    }


def _build_result(session, job) -> dict:
    """Assemble the full output: transcription segments (with speakers), the
    diarization speaker timeline, and the conversation table — all live from the
    persisted segments + diarization artifact."""
    case_id, file_id = job.case_id, job.file_id
    all_segs = repo.list_segments(session, file_id)
    # Reruns add new segments for the same file_id; deduplicate by (start, end)
    # keeping the latest created_at (most recent run wins).
    seen: dict[tuple, object] = {}
    for seg in sorted(all_segs, key=lambda s: s.created_at):
        seen[(round(seg.start, 3), round(seg.end, 3))] = seg
    segs = sorted(seen.values(), key=lambda s: s.start)
    f = repo.get_file(session, file_id)
    src_hash = f.source_sha256 if f else None

    transcript = ts.build(case_id, file_id, src_hash, segs, status=job.status)
    table = ts.build_conversation_table(file_id, segs)

    diar = {"speakers": [], "timeline": [], "model_version": None}
    diar_path = storage.derivative_path(case_id, file_id, "diarization",
                                        f"{file_id}_speaker_timeline.json")
    if os.path.exists(diar_path):
        diar = _json.load(open(diar_path))

    return {
        "job_id": job.id, "case_id": case_id, "file_id": file_id,
        "status": job.status, "source_hash_sha256": src_hash,
        "transcript": transcript,        # segments: start/end/speaker/text/language/confidence
        "diarization": diar,             # speakers + speaker timeline + model_version
        "conversation_table": table,     # court-ready Time/Person/Conversation rows
    }


@router.post("/cases", status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_api_key)])
def create_case():
    with get_session() as s:
        case_id = repo.create_case(s)
        s.commit()
    return {"case_id": case_id}


@router.post("/cases/{case_id}/files", status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(require_api_key)])
async def upload_file(case_id: str, audio: UploadFile = File(...),
                      separate: bool = Form(default=False),
                      num_speakers: int = Form(default=0),
                      enhance_audio: bool = Form(default=False),
                      callback_url: str = Form(default="")):
    ext = os.path.splitext(audio.filename or "")[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    with get_session() as s:
        if s.get(Case, case_id) is None:
            raise HTTPException(status_code=404, detail="Unknown case_id")

    # Create the file row first so we can name the staged upload by file_id;
    # L0 (pipeline) finds it deterministically at cases/{case}/inbox/{file_id}{ext}.
    opts = {"separate": separate}
    if num_speakers and num_speakers > 0:
        opts["num_speakers"] = num_speakers   # exact speaker count hint for diarization
    if enhance_audio:
        opts["enhance_audio"] = True          # force denoised+loudnorm Whisper input (noisy/far-field)
    if callback_url:
        opts["callback_url"] = callback_url   # webhook target for status events
    with get_session() as s:
        file_id = repo.create_file(s, case_id, audio.filename or f"upload{ext}", ext)
        job_id = repo.create_job(s, case_id, file_id, options=opts)
        s.commit()

    inbox = os.path.join(config.CASE_STORE_PATH, "cases", case_id, "inbox")
    os.makedirs(inbox, exist_ok=True)
    staged = os.path.join(inbox, f"{file_id}{ext}")
    content = await audio.read()
    if len(content) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    async with aiofiles.open(staged, "wb") as f:
        await f.write(content)

    run_pipeline.apply_async(args=[job_id], queue=config.GPU_QUEUE)
    return {"file_id": file_id, "job_id": job_id}


@router.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str):
    """Processing status + stage + progress. Poll this, or supply a callback_url
    on upload to receive the same payload as a webhook on every transition."""
    with get_session() as s:
        job = repo.get_job(s, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return _job_view(job)


@router.get("/jobs/{job_id}/result", dependencies=[Depends(require_api_key)])
def get_job_result(job_id: str):
    """Full transcription + diarization (speaker timeline) + court-ready
    conversation table for a finished job. 409 until the job is terminal."""
    with get_session() as s:
        job = repo.get_job(s, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        if job.status not in (JobStatus.COMPLETED, JobStatus.CERTIFIED):
            raise HTTPException(status_code=409,
                                detail=f"job not finished (status={job.status}, stage={job.stage})")
        return _build_result(s, job)


@router.get("/cases/{case_id}/files/{file_id}/result",
            dependencies=[Depends(require_api_key)])
def get_file_result(case_id: str, file_id: str):
    """Same result, addressed by case+file (latest job for that file)."""
    with get_session() as s:
        job = repo.latest_job_for_file(s, file_id)
        if job is None or job.case_id != case_id:
            raise HTTPException(status_code=404, detail="No job for that case/file")
        if job.status not in (JobStatus.COMPLETED, JobStatus.CERTIFIED):
            raise HTTPException(status_code=409,
                                detail=f"job not finished (status={job.status}, stage={job.stage})")
        return _build_result(s, job)


@router.post("/jobs/{job_id}/rerun", status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(require_api_key)])
def rerun_job(job_id: str):
    """Re-process the SAME source file under a NEW job id (e.g. after tuning).
    Re-stages the immutable original; the prior job/segments are untouched."""
    with get_session() as s:
        old = repo.get_job(s, job_id)
        if old is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        f = repo.get_file(s, old.file_id)
        if f is None:
            raise HTTPException(status_code=404, detail="Source file missing")
        case_id, file_id, ext = old.case_id, old.file_id, f.ext
        original = os.path.join(storage.originals_dir(case_id),
                                f"{file_id}__original{ext}")
        if not os.path.exists(original):
            raise HTTPException(status_code=410, detail="Original no longer available")
        # Re-stage original into inbox where L0 expects it.
        inbox = os.path.join(config.CASE_STORE_PATH, "cases", case_id, "inbox")
        os.makedirs(inbox, exist_ok=True)
        shutil.copy(original, os.path.join(inbox, f"{file_id}{ext}"))
        new_job_id = repo.create_job(s, case_id, file_id, options=old.options or {})
        s.commit()
    run_pipeline.apply_async(args=[new_job_id], queue=config.GPU_QUEUE)
    return {"job_id": new_job_id, "case_id": case_id, "file_id": file_id,
            "rerun_of": job_id}


@router.get("/cases/{case_id}", dependencies=[Depends(require_api_key)])
def get_case(case_id: str):
    """List every file + its latest job for a case — lets an app re-hydrate state
    (which jobs exist, their status) when a user reopens the case."""
    with get_session() as s:
        if s.get(Case, case_id) is None:
            raise HTTPException(status_code=404, detail="Unknown case_id")
        files = repo.list_files(s, case_id)
        out = []
        for f in files:
            job = repo.latest_job_for_file(s, f.id)
            out.append({
                "file_id": f.id,
                "original_filename": f.original_filename,
                "latest_job": _job_view(job) if job else None,
            })
        return {"case_id": case_id, "files": out}


@router.post("/cases/{case_id}/files/{file_id}/certify",
             dependencies=[Depends(require_api_key)])
def certify(case_id: str, file_id: str):
    with get_session() as s:
        pending = repo.count_pending_flagged(s, file_id)
        if pending > 0:
            raise HTTPException(status_code=409,
                                detail=f"{pending} flagged segment(s) still pending review")
        # Flip the persisted transcript to certified.
        path = ts.final_path(case_id, file_id)
        if os.path.exists(path):
            data = _json.load(open(path))
            data["status"] = "certified"
            ts.write(case_id, file_id, data)
        job = repo.latest_job_for_file(s, file_id)
        if job is not None:
            repo.update_job(s, job.id, status=JobStatus.CERTIFIED)
        s.commit()
        au.append_entry(case_id, file_id=file_id, stage="certify",
                        parameters={"result": "certified"}, session=s)
        s.commit()
    return {"status": "certified"}
