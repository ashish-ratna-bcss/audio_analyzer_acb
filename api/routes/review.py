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
_VALID_PASSES = {"pass1_whisper", "pass2_indic_conformer", "pass3_seamless"}


class ReviewDecision(BaseModel):
    decision: str
    text: Optional[str] = None
    reviewer_id: str
    selected_pass: Optional[str] = None  # pick specific model transcript on accept


def _case_of(session, file_id):
    f = repo.get_file(session, file_id)
    return f.case_id if f else "unknown"


@router.get("/queue")
def queue(case_id: str, status: str = "pending"):
    with get_session() as s:
        segs = repo.list_flagged_segments(s, case_id, review_status=status)
        return [
            {
                "segment_id": x.id,
                "file_id": x.file_id,
                "start": x.start,
                "end": x.end,
                "speaker": x.speaker,
                "confidence": x.confidence,
                "detected_language": x.detected_language,
                "review_status": x.review_status,
                "current_text": x.text,
                "source_pass": x.source_pass,
                "candidates": x.candidates or {},
            }
            for x in segs
        ]


@router.get("/segments/{segment_id}")
def segment_detail(segment_id: str):
    with get_session() as s:
        seg = repo.get_segment(s, segment_id)
        if seg is None:
            raise HTTPException(status_code=404, detail="Unknown segment_id")
        return {
            "segment_id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "speaker": seg.speaker,
            "detected_language": seg.detected_language,
            "current_text": seg.text,
            "source_pass": seg.source_pass,
            "confidence": seg.confidence,
            "review_status": seg.review_status,
            "candidates": seg.candidates or {},
            "clip_original_url": f"/review/segments/{seg.id}/clip/original",
            "clip_enhanced_url": f"/review/segments/{seg.id}/clip/enhanced",
        }


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
    if body.selected_pass is not None and body.selected_pass not in _VALID_PASSES:
        raise HTTPException(
            status_code=400,
            detail=f"selected_pass must be one of: {', '.join(sorted(_VALID_PASSES))}",
        )
    with get_session() as s:
        seg = repo.get_segment(s, segment_id)
        if seg is None:
            raise HTTPException(status_code=404, detail="Unknown segment_id")
        file_id = seg.file_id
        case_id = _case_of(s, file_id)

        # Resolve winner text: explicit edit > selected_pass > keep current
        final_text = None
        final_source_pass = None
        if body.decision == "edit" and body.text:
            final_text = body.text
            final_source_pass = "human_edited"
        elif body.selected_pass and body.decision == "accept":
            pass_data = (seg.candidates or {}).get(body.selected_pass, {})
            final_text = pass_data.get("text") or seg.text
            final_source_pass = body.selected_pass

        repo.add_review(s, segment_id, body.decision, body.text, body.reviewer_id)
        repo.set_segment_review(
            s, segment_id,
            _DECISION_TO_STATUS[body.decision],
            text=final_text,
            source_pass=final_source_pass,
        )
        s.commit()
        au.append_entry(
            case_id, file_id=file_id, stage="L7_review",
            parameters={
                "segment_id": segment_id,
                "decision": body.decision,
                "selected_pass": body.selected_pass,
                "reviewer_id": body.reviewer_id,
            },
            session=s,
        )
        s.commit()
    return {
        "segment_id": segment_id,
        "review_status": _DECISION_TO_STATUS[body.decision],
        "source_pass": final_source_pass or seg.source_pass,
    }
