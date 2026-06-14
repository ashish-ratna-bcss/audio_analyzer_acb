import os
import uuid
import aiofiles
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional

import config
from models.schemas import TranscribeResponse, Segment
from services.ffmpeg_service import convert_to_wav, UnsupportedFormatError
from services.whisper_service import transcribe
from services.diarization_service import diarize
from services.alignment_service import align_segments
from services.translation_service import translate_segments

router = APIRouter()


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(default="auto"),
    diarize_flag: bool = Form(default=True, alias="diarize"),
    translate: bool = Form(default=False),
    translate_to: str = Form(default="en"),
):
    ext = os.path.splitext(audio.filename or "")[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    job_id = str(uuid.uuid4())
    upload_path = os.path.join(config.UPLOAD_DIR, f"{job_id}{ext}")
    wav_path = os.path.join(config.OUTPUT_DIR, f"{job_id}.wav")

    try:
        async with aiofiles.open(upload_path, "wb") as f:
            content = await audio.read()
            if len(content) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(status_code=413, detail="File too large")
            await f.write(content)

        try:
            convert_to_wav(upload_path, wav_path)
        except UnsupportedFormatError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e}")

        whisper_result = transcribe(wav_path, language=language or "auto")
        whisper_segments = whisper_result["segments"]

        if diarize_flag:
            speaker_segs = diarize(wav_path)
            aligned = align_segments(whisper_segments, speaker_segs)
        else:
            aligned = [
                {**seg, "speaker": "Speaker_1"} for seg in whisper_segments
            ]

        if translate:
            src_lang = "tel_Telu" if whisper_result["language"] == "te" else "eng_Latn"
            tgt_lang = "eng_Latn"
            aligned = translate_segments(aligned, src_lang=src_lang, tgt_lang=tgt_lang)

        full_text = " ".join(seg["text"] for seg in aligned)

        return TranscribeResponse(
            language=whisper_result["language"],
            duration=whisper_result["duration"],
            text=full_text,
            segments=[Segment(**seg) for seg in aligned],
        )

    finally:
        for path in [upload_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)
