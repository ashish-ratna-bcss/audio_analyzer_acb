from pydantic import BaseModel, Field
from typing import Optional, List


class TranscribeRequest(BaseModel):
    language: Optional[str] = Field(default="auto")  # auto-detect or a code like "te", "en"
    debug: bool = False  # include confidence / per-segment metrics in the response


class DialogueTurn(BaseModel):
    start: float
    end: float
    speaker: str
    text: str
    confidence: Optional[float] = None  # populated only when debug=true


class SegmentDetail(BaseModel):
    start: float
    end: float
    speaker: str
    text: str
    confidence: float
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None


class Block(BaseModel):
    """One diarized view of the call: the merged dialogue turns, plus the
    fine-grained segments with review metrics when debug=true."""
    dialogue: List[DialogueTurn]
    segments: Optional[List[SegmentDetail]] = None  # debug only


class TranscribeResponse(BaseModel):
    language: str
    duration: float
    raw: Block        # faithful transcription (spoken language/script), diarized
    english: Block    # Whisper speech->English translation, diarized
