import os
import uuid
import aiofiles
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional

import config
from models.schemas import TranscribeResponse, Block, DialogueTurn, SegmentDetail
from services.ffmpeg_service import convert_to_wav, measure_mean_volume, UnsupportedFormatError
from services.whisper_service import transcribe
from services.diarization_service import diarize
from services.alignment_service import align_segments
from services.dialogue_service import group_turns

router = APIRouter()


def _build_block(whisper_segments: list[dict], speaker_segs: Optional[list[dict]],
                 debug: bool) -> Block:
    """Shape a Whisper pass into a diarized Block. When speaker_segs is given,
    align to the shared speaker timeline; when None (diarization off), put
    everything under a single speaker. Segments + metrics only when debug."""
    if speaker_segs is not None:
        aligned = align_segments(whisper_segments, speaker_segs)
    else:
        aligned = [{**s, "speaker": "Speaker_1"} for s in whisper_segments]
    turns = group_turns(aligned)

    dialogue = [
        DialogueTurn(
            start=t["start"], end=t["end"], speaker=t["speaker"], text=t["text"],
            confidence=t["confidence"] if debug else None,
        )
        for t in turns
    ]

    segments = None
    if debug:
        segments = [
            SegmentDetail(
                start=s["start"], end=s["end"], speaker=s["speaker"], text=s["text"],
                confidence=s["confidence"],
                no_speech_prob=s.get("no_speech_prob"),
                compression_ratio=s.get("compression_ratio"),
            )
            for s in aligned
        ]

    return Block(dialogue=dialogue, segments=segments)


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    response_model_exclude_none=True,  # drop None confidence/segments in default mode
)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(default="auto"),
    diarize_flag: bool = Form(default=True, alias="diarize"),
    debug: bool = Form(default=False),
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

        # Dynamic VAD: very low-volume audio gets VAD off (else VAD can strip it
        # entirely); normal audio keeps least-aggressive VAD.
        mean_db = measure_mean_volume(wav_path)
        use_vad = not (mean_db is not None and mean_db < config.VAD_MIN_MEAN_DB)

        # Diarization is on by default (pass diarize=false to disable). It runs
        # on the audio (language-independent), so run it ONCE and align both
        # Whisper passes to the same speaker timeline -> speaker labels stay
        # consistent across the raw and English views.
        speaker_segs = diarize(wav_path) if diarize_flag else None

        lang = language or "auto"
        raw_result = transcribe(wav_path, language=lang, use_vad=use_vad, task="transcribe")
        en_result = transcribe(wav_path, language=lang, use_vad=use_vad, task="translate")

        return TranscribeResponse(
            language=raw_result["language"],
            duration=raw_result["duration"],
            raw=_build_block(raw_result["segments"], speaker_segs, debug),
            english=_build_block(en_result["segments"], speaker_segs, debug),
        )

    finally:
        for path in [upload_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)
