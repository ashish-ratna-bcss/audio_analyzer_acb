import os

import aiofiles
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, status

import config
from api.auth import require_api_key
from db.base import get_session
from db import repository as repo
from db.models import Case
from pipeline.tasks import run_pipeline

router = APIRouter()


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
                      separate: bool = Form(default=False)):
    ext = os.path.splitext(audio.filename or "")[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    with get_session() as s:
        if s.get(Case, case_id) is None:
            raise HTTPException(status_code=404, detail="Unknown case_id")

    # Create the file row first so we can name the staged upload by file_id;
    # L0 (pipeline) finds it deterministically at cases/{case}/inbox/{file_id}{ext}.
    with get_session() as s:
        file_id = repo.create_file(s, case_id, audio.filename or f"upload{ext}", ext)
        job_id = repo.create_job(s, case_id, file_id, options={"separate": separate})
        s.commit()

    inbox = os.path.join(config.CASE_STORE_PATH, "cases", case_id, "inbox")
    os.makedirs(inbox, exist_ok=True)
    staged = os.path.join(inbox, f"{file_id}{ext}")
    content = await audio.read()
    if len(content) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    async with aiofiles.open(staged, "wb") as f:
        await f.write(content)

    run_pipeline.delay(job_id)
    return {"file_id": file_id, "job_id": job_id}


@router.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str):
    with get_session() as s:
        job = repo.get_job(s, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return {
            "job_id": job.id,
            "case_id": job.case_id,
            "file_id": job.file_id,
            "status": job.status,
            "stage": job.stage,
            "degraded_flags": job.degraded_flags or [],
        }
