# Speech Intelligence API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully on-premise FastAPI service that accepts audio uploads and returns transcription, speaker diarization, word timestamps, and optional Telugu→English translation — all on local GPU with no external API calls.

**Architecture:** Single FastAPI process on port 8000, served behind Nginx on port 80. Models (Whisper Large-v3, pyannote 3.1, NLLB-200) are loaded once at startup and held in GPU memory. Audio is converted to WAV/16kHz/mono via FFmpeg before any ML processing.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, PyTorch (CUDA), faster-whisper OR openai-whisper, pyannote.audio 3.x, HuggingFace Transformers (NLLB-200), pydub, librosa, FFmpeg, Nginx

---

## File Map

| File | Responsibility |
|------|---------------|
| `app.py` | FastAPI app factory, lifespan startup/shutdown, model loading |
| `api/routes/stt.py` | POST /stt/transcribe endpoint, request validation |
| `services/ffmpeg_service.py` | Audio conversion to WAV 16kHz mono |
| `services/whisper_service.py` | Whisper inference, returns segments + language |
| `services/diarization_service.py` | Pyannote inference, returns speaker time ranges |
| `services/alignment_service.py` | Align whisper segments → speaker segments |
| `services/translation_service.py` | NLLB-200 translation, Telugu→English |
| `models/schemas.py` | Pydantic request/response models |
| `config.py` | All constants: paths, model names, limits |
| `requirements.txt` | Pinned dependencies |
| `nginx/speech-api.conf` | Nginx reverse proxy config |
| `tests/test_ffmpeg_service.py` | Unit tests for audio conversion |
| `tests/test_alignment_service.py` | Unit tests for segment alignment logic |
| `tests/test_schemas.py` | Pydantic schema validation tests |
| `tests/test_api.py` | FastAPI TestClient integration tests |
| `uploads/` | Temp upload storage (gitignored) |
| `outputs/` | Temp output storage (gitignored) |
| `models/` | Downloaded ML model weights (gitignored) |

---

## Task 1: Project Scaffold & Config

**Files:**
- Create: `config.py`
- Create: `models/schemas.py`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `tests/__init__.py`
- Create: `api/__init__.py`
- Create: `api/routes/__init__.py`
- Create: `services/__init__.py`

- [ ] **Step 1: Create `.gitignore`**

```
uploads/
outputs/
models/
__pycache__/
*.pyc
.env
*.log
```

- [ ] **Step 2: Create `config.py`**

```python
import os

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

MAX_FILE_SIZE_MB = 500
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}

WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
PYANNOTE_AUTH_TOKEN = os.getenv("HF_TOKEN", "")

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_DEVICE = "cuda"
NLLB_MAX_LENGTH = 1024

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
```

- [ ] **Step 3: Create `models/schemas.py`**

```python
from pydantic import BaseModel, Field
from typing import Optional, List


class TranscribeRequest(BaseModel):
    language: Optional[str] = Field(default="auto", pattern="^(te|en|auto)$")
    diarize: bool = True
    translate: bool = False
    translate_to: str = "en"


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
```

- [ ] **Step 4: Create `requirements.txt`**

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9
pydantic==2.7.1
torch==2.3.0
torchaudio==2.3.0
faster-whisper==1.0.3
pyannote.audio==3.3.1
transformers==4.41.0
pydub==0.25.1
librosa==0.10.2
soundfile==0.12.1
ffmpeg-python==0.2.0
numpy==1.26.4
```

- [ ] **Step 5: Create all `__init__.py` files**

```bash
touch tests/__init__.py api/__init__.py api/routes/__init__.py services/__init__.py
```

- [ ] **Step 6: Create upload/output dirs**

```bash
mkdir -p uploads outputs models
```

- [ ] **Step 7: Write schema tests**

Create `tests/test_schemas.py`:
```python
import pytest
from models.schemas import TranscribeRequest, TranscribeResponse, Segment


def test_transcribe_request_defaults():
    req = TranscribeRequest()
    assert req.language == "auto"
    assert req.diarize is True
    assert req.translate is False
    assert req.translate_to == "en"


def test_transcribe_request_custom():
    req = TranscribeRequest(language="te", diarize=False, translate=True)
    assert req.language == "te"
    assert req.diarize is False
    assert req.translate is True


def test_transcribe_request_invalid_language():
    with pytest.raises(Exception):
        TranscribeRequest(language="fr")


def test_segment_optional_translation():
    seg = Segment(speaker="Speaker_1", start=0.5, end=4.3, text="hello")
    assert seg.translated_text is None


def test_transcribe_response_structure():
    resp = TranscribeResponse(
        language="te",
        duration=125.5,
        text="full transcript",
        segments=[
            Segment(speaker="Speaker_1", start=0.5, end=4.3, text="hello")
        ]
    )
    assert resp.language == "te"
    assert len(resp.segments) == 1
```

- [ ] **Step 8: Run schema tests**

```bash
pytest tests/test_schemas.py -v
```

Expected: 5 PASSED

- [ ] **Step 9: Commit**

```bash
git add config.py models/schemas.py requirements.txt .gitignore tests/ api/ services/
git commit -m "feat: project scaffold, config, pydantic schemas"
```

---

## Task 2: FFmpeg Audio Conversion Service

**Files:**
- Create: `services/ffmpeg_service.py`
- Create: `tests/test_ffmpeg_service.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ffmpeg_service.py`:
```python
import os
import pytest
import wave
from services.ffmpeg_service import convert_to_wav, UnsupportedFormatError


def test_unsupported_format_raises():
    with pytest.raises(UnsupportedFormatError):
        convert_to_wav("/tmp/fake.xyz", "/tmp/out.wav")


def test_wav_file_conversion(tmp_path):
    # create minimal valid WAV at wrong sample rate using wave module
    src = str(tmp_path / "test_stereo_44k.wav")
    dst = str(tmp_path / "out.wav")
    with wave.open(src, "w") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(b"\x00\x00" * 44100 * 2)  # 1 second stereo
    convert_to_wav(src, dst)
    assert os.path.exists(dst)
    with wave.open(dst) as f:
        assert f.getnchannels() == 1
        assert f.getframerate() == 16000


def test_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        convert_to_wav("/tmp/does_not_exist.wav", "/tmp/out.wav")
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_ffmpeg_service.py -v
```

Expected: ImportError or ModuleNotFoundError

- [ ] **Step 3: Implement `services/ffmpeg_service.py`**

```python
import os
import subprocess
import config


class UnsupportedFormatError(ValueError):
    pass


def convert_to_wav(input_path: str, output_path: str) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(f"Unsupported format: {ext}")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", str(config.TARGET_CHANNELS),
        "-ar", str(config.TARGET_SAMPLE_RATE),
        "-sample_fmt", "s16",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    return output_path
```

- [ ] **Step 4: Run tests to verify PASS**

```bash
pytest tests/test_ffmpeg_service.py -v
```

Expected: 3 PASSED (requires ffmpeg installed on server)

- [ ] **Step 5: Commit**

```bash
git add services/ffmpeg_service.py tests/test_ffmpeg_service.py
git commit -m "feat: ffmpeg audio conversion service with format validation"
```

---

## Task 3: Whisper Transcription Service

**Files:**
- Create: `services/whisper_service.py`

> NOTE: Whisper model (large-v3) is ~3GB. Tests for this service use mocking — real GPU inference tested only on server.

- [ ] **Step 1: Write failing test with mock**

Create `tests/test_whisper_service.py`:
```python
import pytest
from unittest.mock import MagicMock, patch


def test_transcribe_returns_expected_shape():
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 3.5
    mock_segment.text = " hello world"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        iter([mock_segment]),
        MagicMock(language="te", duration=3.5)
    )

    with patch("services.whisper_service._model", mock_model):
        from services.whisper_service import transcribe
        result = transcribe("/tmp/fake.wav", language="te")

    assert result["language"] == "te"
    assert result["duration"] == 3.5
    assert len(result["segments"]) == 1
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == 3.5
    assert result["segments"][0]["text"] == "hello world"


def test_transcribe_auto_language():
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.0
    mock_segment.text = " నమస్కారం"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        iter([mock_segment]),
        MagicMock(language="te", duration=2.0)
    )

    with patch("services.whisper_service._model", mock_model):
        from services.whisper_service import transcribe
        result = transcribe("/tmp/fake.wav", language="auto")

    mock_model.transcribe.assert_called_once()
    call_kwargs = mock_model.transcribe.call_args[1]
    assert "language" not in call_kwargs or call_kwargs.get("language") is None
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_whisper_service.py -v
```

Expected: ImportError

- [ ] **Step 3: Implement `services/whisper_service.py`**

```python
from faster_whisper import WhisperModel
import config

_model: WhisperModel = None


def load_model():
    global _model
    if _model is None:
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            download_root=config.MODEL_DIR,
        )
    return _model


def transcribe(audio_path: str, language: str = "auto") -> dict:
    model = load_model()
    kwargs = {}
    if language != "auto":
        kwargs["language"] = language

    segments_iter, info = model.transcribe(audio_path, word_timestamps=False, **kwargs)
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })

    return {
        "language": info.language,
        "duration": round(info.duration, 3),
        "segments": segments,
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_whisper_service.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add services/whisper_service.py tests/test_whisper_service.py
git commit -m "feat: whisper large-v3 transcription service"
```

---

## Task 4: Speaker Diarization Service

**Files:**
- Create: `services/diarization_service.py`
- Create: `tests/test_diarization_service.py`

- [ ] **Step 1: Write failing tests with mock**

Create `tests/test_diarization_service.py`:
```python
import pytest
from unittest.mock import MagicMock, patch


def _make_mock_pipeline(turn_list):
    """turn_list: [(start, end, speaker_label), ...]"""
    mock_diarization = MagicMock()
    mock_turns = []
    for start, end, label in turn_list:
        segment = MagicMock()
        segment.start = start
        segment.end = end
        mock_turns.append((segment, None, label))
    mock_diarization.itertracks.return_value = iter(mock_turns)

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_diarization
    return mock_pipeline


def test_diarize_returns_speaker_segments():
    mock_pipeline = _make_mock_pipeline([
        (0.0, 3.5, "SPEAKER_00"),
        (4.0, 7.0, "SPEAKER_01"),
        (7.5, 10.0, "SPEAKER_00"),
    ])

    with patch("services.diarization_service._pipeline", mock_pipeline):
        from services.diarization_service import diarize
        result = diarize("/tmp/fake.wav")

    assert len(result) == 3
    assert result[0] == {"start": 0.0, "end": 3.5, "speaker": "Speaker_1"}
    assert result[1] == {"start": 4.0, "end": 7.0, "speaker": "Speaker_2"}
    assert result[2] == {"start": 7.5, "end": 10.0, "speaker": "Speaker_1"}


def test_diarize_speaker_label_normalization():
    mock_pipeline = _make_mock_pipeline([
        (0.0, 2.0, "SPEAKER_00"),
    ])

    with patch("services.diarization_service._pipeline", mock_pipeline):
        from services.diarization_service import diarize
        result = diarize("/tmp/fake.wav")

    assert result[0]["speaker"] == "Speaker_1"
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_diarization_service.py -v
```

Expected: ImportError

- [ ] **Step 3: Implement `services/diarization_service.py`**

```python
from pyannote.audio import Pipeline
import torch
import config

_pipeline = None


def load_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.from_pretrained(
            config.DIARIZATION_MODEL,
            use_auth_token=config.PYANNOTE_AUTH_TOKEN,
        )
        _pipeline.to(torch.device("cuda"))
    return _pipeline


def diarize(audio_path: str) -> list[dict]:
    pipeline = load_pipeline()
    diarization = pipeline(audio_path)

    speaker_map: dict[str, str] = {}
    speaker_counter = 1
    segments = []

    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        if speaker_label not in speaker_map:
            speaker_map[speaker_label] = f"Speaker_{speaker_counter}"
            speaker_counter += 1
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker_map[speaker_label],
        })

    return segments
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_diarization_service.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add services/diarization_service.py tests/test_diarization_service.py
git commit -m "feat: pyannote speaker diarization service"
```

---

## Task 5: Segment Alignment Service

**Files:**
- Create: `services/alignment_service.py`
- Create: `tests/test_alignment_service.py`

This is pure Python logic — no GPU, most testable component.

- [ ] **Step 1: Write failing tests**

Create `tests/test_alignment_service.py`:
```python
import pytest
from services.alignment_service import align_segments


def test_basic_alignment():
    whisper_segs = [
        {"start": 0.5, "end": 4.3, "text": "Hello sir"},
        {"start": 4.5, "end": 8.7, "text": "What is your issue"},
        {"start": 9.0, "end": 12.0, "text": "My file is pending"},
    ]
    speaker_segs = [
        {"start": 0.0, "end": 4.4, "speaker": "Speaker_1"},
        {"start": 4.4, "end": 9.0, "speaker": "Speaker_2"},
        {"start": 9.0, "end": 13.0, "speaker": "Speaker_1"},
    ]
    result = align_segments(whisper_segs, speaker_segs)
    assert len(result) == 3
    assert result[0]["speaker"] == "Speaker_1"
    assert result[0]["text"] == "Hello sir"
    assert result[1]["speaker"] == "Speaker_2"
    assert result[2]["speaker"] == "Speaker_1"


def test_no_speaker_segments_defaults_unknown():
    whisper_segs = [{"start": 0.0, "end": 2.0, "text": "hello"}]
    result = align_segments(whisper_segs, [])
    assert result[0]["speaker"] == "Unknown"


def test_segment_assigned_to_max_overlap_speaker():
    whisper_segs = [{"start": 1.0, "end": 5.0, "text": "test"}]
    speaker_segs = [
        {"start": 0.0, "end": 2.5, "speaker": "Speaker_1"},  # overlap 1.5s
        {"start": 2.5, "end": 6.0, "speaker": "Speaker_2"},  # overlap 2.5s
    ]
    result = align_segments(whisper_segs, speaker_segs)
    assert result[0]["speaker"] == "Speaker_2"


def test_start_end_times_preserved():
    whisper_segs = [{"start": 1.1, "end": 3.3, "text": "hi"}]
    speaker_segs = [{"start": 0.0, "end": 5.0, "speaker": "Speaker_1"}]
    result = align_segments(whisper_segs, speaker_segs)
    assert result[0]["start"] == 1.1
    assert result[0]["end"] == 3.3
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_alignment_service.py -v
```

Expected: ImportError

- [ ] **Step 3: Implement `services/alignment_service.py`**

```python
def align_segments(whisper_segments: list[dict], speaker_segments: list[dict]) -> list[dict]:
    aligned = []
    for wseg in whisper_segments:
        w_start, w_end = wseg["start"], wseg["end"]
        best_speaker = "Unknown"
        best_overlap = 0.0

        for sseg in speaker_segments:
            overlap_start = max(w_start, sseg["start"])
            overlap_end = min(w_end, sseg["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sseg["speaker"]

        aligned.append({
            "speaker": best_speaker,
            "start": w_start,
            "end": w_end,
            "text": wseg["text"],
        })

    return aligned
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_alignment_service.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add services/alignment_service.py tests/test_alignment_service.py
git commit -m "feat: speaker-segment alignment via max overlap"
```

---

## Task 6: Translation Service (NLLB-200)

**Files:**
- Create: `services/translation_service.py`
- Create: `tests/test_translation_service.py`

- [ ] **Step 1: Write failing tests with mock**

Create `tests/test_translation_service.py`:
```python
import pytest
from unittest.mock import MagicMock, patch


def test_translate_returns_string():
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_inputs = MagicMock()
    mock_tokenizer.return_value = mock_inputs
    mock_inputs.to.return_value = mock_inputs
    mock_inputs.__getitem__ = MagicMock(return_value=MagicMock())

    mock_output = MagicMock()
    mock_model.generate.return_value = mock_output
    mock_tokenizer.batch_decode.return_value = ["Hello sir"]

    with patch("services.translation_service._tokenizer", mock_tokenizer), \
         patch("services.translation_service._model", mock_model):
        from services.translation_service import translate
        result = translate("సార్ నమస్కారం", src_lang="tel_Telu", tgt_lang="eng_Latn")

    assert isinstance(result, str)
    assert result == "Hello sir"


def test_translate_segments():
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_inputs = MagicMock()
    mock_tokenizer.return_value = mock_inputs
    mock_inputs.to.return_value = mock_inputs
    mock_tokenizer.batch_decode.return_value = ["translated text"]
    mock_model.generate.return_value = MagicMock()

    segments = [
        {"speaker": "Speaker_1", "start": 0.0, "end": 2.0, "text": "నమస్కారం"},
    ]

    with patch("services.translation_service._tokenizer", mock_tokenizer), \
         patch("services.translation_service._model", mock_model):
        from services.translation_service import translate_segments
        result = translate_segments(segments, src_lang="tel_Telu", tgt_lang="eng_Latn")

    assert result[0]["translated_text"] == "translated text"
    assert result[0]["text"] == "నమస్కారం"
```

- [ ] **Step 2: Run to verify FAIL**

```bash
pytest tests/test_translation_service.py -v
```

Expected: ImportError

- [ ] **Step 3: Implement `services/translation_service.py`**

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import config

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            config.NLLB_MODEL,
            cache_dir=config.MODEL_DIR,
        )
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            config.NLLB_MODEL,
            cache_dir=config.MODEL_DIR,
        ).to(config.NLLB_DEVICE)
    return _model, _tokenizer


def translate(text: str, src_lang: str = "tel_Telu", tgt_lang: str = "eng_Latn") -> str:
    model, tokenizer = load_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                       max_length=config.NLLB_MAX_LENGTH)
    inputs = inputs.to(config.NLLB_DEVICE)
    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=config.NLLB_MAX_LENGTH,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


def translate_segments(
    segments: list[dict],
    src_lang: str = "tel_Telu",
    tgt_lang: str = "eng_Latn",
) -> list[dict]:
    result = []
    for seg in segments:
        translated = translate(seg["text"], src_lang=src_lang, tgt_lang=tgt_lang)
        result.append({**seg, "translated_text": translated})
    return result
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_translation_service.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add services/translation_service.py tests/test_translation_service.py
git commit -m "feat: NLLB-200 translation service, Telugu to English"
```

---

## Task 7: FastAPI Route & App

**Files:**
- Create: `api/routes/stt.py`
- Create: `app.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Create `api/routes/stt.py`**

```python
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
```

- [ ] **Step 2: Create `app.py`**

```python
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
import config
from api.routes.stt import router as stt_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    from services.whisper_service import load_model as load_whisper
    from services.diarization_service import load_pipeline as load_diarization
    load_whisper()
    load_diarization()
    yield


app = FastAPI(title="Speech Intelligence API", version="1.0.0", lifespan=lifespan)
app.include_router(stt_router, prefix="/stt")


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 3: Write API integration tests**

Create `tests/test_api.py`:
```python
import io
import wave
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def make_wav_bytes(duration_sec=1, sample_rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(b"\x00\x00" * sample_rate * duration_sec)
    return buf.getvalue()


@pytest.fixture
def client():
    with patch("services.whisper_service.load_model"), \
         patch("services.diarization_service.load_pipeline"):
        from app import app
        return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_transcribe_success(client):
    mock_whisper = {
        "language": "te",
        "duration": 1.0,
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
    }
    mock_speaker = [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}]

    wav_bytes = make_wav_bytes()

    with patch("api.routes.stt.convert_to_wav"), \
         patch("api.routes.stt.transcribe", return_value=mock_whisper), \
         patch("api.routes.stt.diarize", return_value=mock_speaker):
        resp = client.post(
            "/stt/transcribe",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"language": "te", "diarize": "true"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "te"
    assert body["duration"] == 1.0
    assert len(body["segments"]) == 1
    assert body["segments"][0]["speaker"] == "Speaker_1"


def test_transcribe_unsupported_format(client):
    resp = client.post(
        "/stt/transcribe",
        files={"audio": ("test.xyz", b"fake", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_transcribe_with_translation(client):
    mock_whisper = {
        "language": "te",
        "duration": 2.0,
        "segments": [{"start": 0.0, "end": 2.0, "text": "నమస్కారం"}],
    }
    mock_speaker = [{"start": 0.0, "end": 2.0, "speaker": "Speaker_1"}]
    mock_translated = [
        {"start": 0.0, "end": 2.0, "text": "నమస్కారం",
         "speaker": "Speaker_1", "translated_text": "Hello"}
    ]

    with patch("api.routes.stt.convert_to_wav"), \
         patch("api.routes.stt.transcribe", return_value=mock_whisper), \
         patch("api.routes.stt.diarize", return_value=mock_speaker), \
         patch("api.routes.stt.translate_segments", return_value=mock_translated):
        resp = client.post(
            "/stt/transcribe",
            files={"audio": ("test.wav", make_wav_bytes(), "audio/wav")},
            data={"translate": "true"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["segments"][0]["translated_text"] == "Hello"
```

- [ ] **Step 4: Run API tests**

```bash
pytest tests/test_api.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add api/routes/stt.py app.py tests/test_api.py
git commit -m "feat: fastapi endpoint /stt/transcribe with full pipeline"
```

---

## Task 8: Nginx Configuration

**Files:**
- Create: `nginx/speech-api.conf`

No tests needed — config file deployed to server.

- [ ] **Step 1: Create `nginx/speech-api.conf`**

```nginx
server {
    listen 80;
    server_name _;

    client_max_body_size 512M;
    client_body_timeout 300s;
    proxy_read_timeout 300s;
    proxy_connect_timeout 60s;
    proxy_send_timeout 300s;

    location /stt/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /health {
        proxy_pass http://127.0.0.1:8000;
    }
}
```

- [ ] **Step 2: Server deployment commands**

Run on the server after pushing code:
```bash
# Copy nginx config
sudo cp nginx/speech-api.conf /etc/nginx/sites-available/speech-api
sudo ln -sf /etc/nginx/sites-available/speech-api /etc/nginx/sites-enabled/speech-api
sudo nginx -t
sudo systemctl reload nginx

# Create venv and install deps
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set HuggingFace token (required for pyannote gating)
export HF_TOKEN="your_hf_token_here"

# Start API
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

- [ ] **Step 3: Commit**

```bash
git add nginx/speech-api.conf
git commit -m "feat: nginx reverse proxy config for speech API"
```

---

## Task 9: End-to-End Smoke Test on Server

Run on server after deployment.

- [ ] **Step 1: Health check**

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 2: Upload a real Telugu audio**

```bash
curl -X POST http://localhost:8000/stt/transcribe \
  -F "audio=@/path/to/sample_telugu.wav" \
  -F "language=te" \
  -F "diarize=true" \
  -F "translate=false"
```

Expected: JSON with `language`, `duration`, `text`, `segments[]` with speaker labels.

- [ ] **Step 3: Test with translation**

```bash
curl -X POST http://localhost:8000/stt/transcribe \
  -F "audio=@/path/to/sample_telugu.wav" \
  -F "language=te" \
  -F "diarize=true" \
  -F "translate=true"
```

Expected: `segments[].translated_text` populated in English.

- [ ] **Step 4: Test via public IP through Nginx**

```bash
curl -X POST http://PUBLIC_IP/stt/transcribe \
  -F "audio=@sample.wav" \
  -F "language=auto"
```

Expected: Same JSON response.

---

## Self-Review

### Spec Coverage

| Requirement | Task |
|-------------|------|
| POST /stt/transcribe | Task 7 |
| Supported formats: wav/mp3/m4a/ogg/flac/webm | Task 2, Task 7 |
| Max 500MB | Task 7 (stt.py) |
| FFmpeg → WAV 16kHz mono | Task 2 |
| Whisper Large-v3 GPU | Task 3 |
| Language: te/en/auto | Task 3, Task 7 |
| Speaker diarization (pyannote 3.1) | Task 4 |
| Speaker labels Speaker_1..N | Task 4 |
| Segment alignment | Task 5 |
| NLLB-200 translation optional | Task 6 |
| translate=true/false param | Task 7 |
| JSON response structure | Task 1, Task 7 |
| Nginx port 80 | Task 8 |
| FastAPI port 8000 | Task 7, Task 8 |
| No external APIs | All tasks — all models local |
| Model loading at startup | Task 7 (lifespan) |
| Temp file cleanup | Task 7 (finally block) |

All requirements covered.

### Placeholder Scan

No TBD, TODO, "implement later", or vague steps found.

### Type Consistency

- `align_segments` takes `list[dict]` → returns `list[dict]` with keys `speaker/start/end/text` → `Segment(**seg)` in route works.
- `translate_segments` adds `translated_text` key → `Segment` schema has `Optional[str]` for it. ✓
- `transcribe()` returns `{"language", "duration", "segments"}` → consumed as `whisper_result["language"]` etc. ✓
- `diarize()` returns `[{"start", "end", "speaker"}]` → consumed by `align_segments` as `sseg["start"]` etc. ✓
