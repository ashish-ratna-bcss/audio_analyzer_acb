# Dual-Output Transcription (raw + English) — Design

Date: 2026-06-16

## Goal

Every audio uploaded to `POST /stt/transcribe` returns, in clean JSON, two
diarized conversational views:

1. **raw** — faithful transcription in the spoken language/script (code-switched
   Telugu+English as actually spoken).
2. **english** — English translation produced by Whisper's built-in translate
   task (speech -> English).

For ACB trap-case evidence: accuracy and completeness first; readable English
plus the verbatim raw, side by side.

## Decisions

- **English via Whisper `task=translate`**, not NLLB. Whisper translates speech
  directly to English — higher quality on code-switched audio, no
  source-language assumption. (NLLB path retired; `translation_service.py` left
  in tree but unused.)
- **Two separate diarized blocks** (`raw`, `english`), each self-consistent.
  No cross-pass per-turn alignment; the two are independently segmented.
- **Diarization once.** pyannote runs on the audio (language-independent); both
  Whisper passes align to the same speaker timeline -> consistent speakers.
- **Both always produced.** Old `translate` / `translate_to` flags removed.
  Params: `audio`, `language` (default auto), `diarize` (default true; pass
  false to disable speaker separation -> single Speaker_1), `debug`.
- **`debug` flag.** Default response is minimal per turn (`start`, `end`,
  `speaker`, `text`). `debug=true` adds per-turn `confidence` and a per-block
  `segments[]` array with `confidence`, `no_speech_prob`, `compression_ratio`.

## Data flow

```
upload
 -> ffmpeg convert (-vn, map 0:a:0?, pcm_s16le, aresample async)
 -> measure_mean_volume -> dynamic use_vad
 -> diarize(wav)                         # once -> speaker_segs
 -> transcribe(task="transcribe")        # raw segments
 -> transcribe(task="translate")         # english segments
 -> each: align_segments(speaker_segs) -> group_turns -> Block
 -> TranscribeResponse{ language, duration, raw, english }
```

## Response shape

Default:
```json
{
  "language": "te",
  "duration": 58.069,
  "raw":     { "dialogue": [ { "start":1.46,"end":9.67,"speaker":"Speaker_1","text":"హలో good evening sir..." } ] },
  "english": { "dialogue": [ { "start":1.46,"end":9.67,"speaker":"Speaker_1","text":"Hello good evening sir..." } ] }
}
```
`debug=true` adds `confidence` to each turn and a `segments[]` array per block.

## Components changed

- `services/whisper_service.py` — `transcribe(..., task=...)`.
- `models/schemas.py` — `DialogueTurn`, `SegmentDetail`, `Block`,
  `TranscribeResponse{language,duration,raw,english}`, `TranscribeRequest{language,debug}`.
- `api/routes/stt.py` — diarize once, two passes, `_build_block`, debug gating,
  `response_model_exclude_none=True`.
- Tests updated (`tests/test_api.py`, `tests/test_schemas.py`).

## Known limits / out of scope

- Two Whisper passes + diarization are synchronous. For ~1-4 min calls this is
  within nginx's 300s timeout; very long files (>~8-10 min) may time out. An
  async job queue is out of scope (YAGNI) until needed.
- ASR is not verbatim-perfect on code-switched Telugu phone audio; human review
  against source audio remains required (see FORENSIC_NOTES.md).
