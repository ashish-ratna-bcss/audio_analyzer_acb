from pydantic import BaseModel, Field
from typing import Optional, List


class TranscribeRequest(BaseModel):
    language: Optional[str] = Field(default="auto")  # Auto-detect or specify language code (e.g., "te", "en", "hi", etc.)
    diarize: bool = True
    translate: bool = False
    translate_to: str = Field(default="en")  # Currently supports "en", extensible for other NLLB targets


class Segment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str
    confidence: float = 0.5  # 0.0-1.0, lower = less confident
    translated_text: Optional[str] = None


class TranscribeResponse(BaseModel):
    language: str
    duration: float
    text: str
    segments: List[Segment]
