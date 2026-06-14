from pydantic import BaseModel, Field
from typing import Optional, List


class TranscribeRequest(BaseModel):
    language: Optional[str] = Field(default="auto", pattern="^(te|en|auto)$")
    diarize: bool = True
    translate: bool = False
    translate_to: str = Field(default="en", pattern="^en$")


class Segment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str
    translated_text: Optional[str] = None


class TranscribeResponse(BaseModel):
    language: str
    duration: float
    text: str
    segments: List[Segment]
