import os

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from api.auth import require_api_key
from db.base import get_session
from db import repository as repo
from services import audit_service as au

router = APIRouter(prefix="/review", dependencies=[Depends(require_api_key)])

_DECISION_TO_STATUS = {"accept": "accepted", "edit": "edited", "reject": "rejected"}


class ReviewDecision(BaseModel):
    decision: str
    text: Optional[str] = None
    reviewer_id: str


def _case_of(session, file_id):
    f = repo.get_file(session, file_id)
    return f.case_id if f else "unknown"


@router.get("/queue")
def queue(case_id: str, status: str = "pending"):
    with get_session() as s:
        segs = repo.list_flagged_segments(s, case_id, review_status=status)
        return [{"segment_id": x.id, "file_id": x.file_id, "start": x.start,
                 "end": x.end, "speaker": x.speaker, "confidence": x.confidence,
                 "review_status": x.review_status} for x in segs]


@router.get("/segments/{segment_id}")
def segment_detail(segment_id: str):
    with get_session() as s:
        seg = repo.get_segment(s, segment_id)
        if seg is None:
            raise HTTPException(status_code=404, detail="Unknown segment_id")
        return {"segment_id": seg.id, "candidates": seg.candidates or {},
                "clip_original_url": f"/review/segments/{seg.id}/clip/original",
                "clip_enhanced_url": f"/review/segments/{seg.id}/clip/enhanced",
                "confidence": seg.confidence, "review_status": seg.review_status,
                "text": seg.text}


@router.get("/segments/{segment_id}/clip/{which}")
def segment_clip(segment_id: str, which: str):
    with get_session() as s:
        seg = repo.get_segment(s, segment_id)
        if seg is None:
            raise HTTPException(status_code=404, detail="Unknown segment_id")
        path = seg.clip_enhanced if which == "enhanced" else seg.clip_original
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Clip not available")
    return FileResponse(path, media_type="audio/wav")


@router.post("/segments/{segment_id}")
def submit_decision(segment_id: str, body: ReviewDecision):
    if body.decision not in _DECISION_TO_STATUS:
        raise HTTPException(status_code=400, detail="decision must be accept|edit|reject")
    with get_session() as s:
        seg = repo.get_segment(s, segment_id)
        if seg is None:
            raise HTTPException(status_code=404, detail="Unknown segment_id")
        file_id = seg.file_id
        case_id = _case_of(s, file_id)
        new_text = body.text if body.decision == "edit" else None
        repo.add_review(s, segment_id, body.decision, body.text, body.reviewer_id)
        repo.set_segment_review(s, segment_id, _DECISION_TO_STATUS[body.decision],
                                text=new_text)
        s.commit()
        au.append_entry(case_id, file_id=file_id, stage="L7_review",
                        parameters={"segment_id": segment_id,
                                    "decision": body.decision,
                                    "reviewer_id": body.reviewer_id}, session=s)
        s.commit()
    return {"segment_id": segment_id, "review_status": _DECISION_TO_STATUS[body.decision]}
