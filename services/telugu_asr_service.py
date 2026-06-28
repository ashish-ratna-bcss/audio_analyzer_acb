"""Language-fine-tuned Whisper (default Telugu: vasista22/whisper-telugu-large-v2).

Third ASR engine. Generic Whisper-large-v3 mishears fast/noisy native Telugu;
this fine-tune is far more accurate on pure-native speech (but transliterates
code-mixed English, so the selector only routes pure-native turns here). Loaded
via transformers (downloads to MODEL_DIR like IndicConformer / MMS-LID) — no CT2
conversion. Whole-file decode with word timestamps mirrors whisper_service so the
caller slices per turn identically.
"""
import logging

import config

logger = logging.getLogger(__name__)

_pipe = None


def _load():
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline
        device = 0 if config.WHISPER_DEVICE == "cuda" else -1
        dtype = torch.float16 if config.WHISPER_DEVICE == "cuda" else torch.float32
        logger.info("Loading fine-tuned Whisper (%s)…", config.WHISPER_TELUGU_MODEL)
        _pipe = pipeline(
            task="automatic-speech-recognition",
            model=config.WHISPER_TELUGU_MODEL,
            chunk_length_s=30,            # long-audio chunking for whole-file decode
            device=device,
            torch_dtype=dtype,
            model_kwargs={"cache_dir": config.MODEL_DIR},
        )
        # The fine-tune ships an incomplete generation_config (missing
        # no_timestamps_token_id + alignment_heads), so return_timestamps="word"
        # raises. Attach the base whisper-large-v2 generation config, which has
        # the timestamp tokens and cross-attention alignment heads needed for
        # word-level DTW timestamps. Keep model-specific fields where present.
        try:
            from transformers import GenerationConfig
            base = GenerationConfig.from_pretrained("openai/whisper-large-v2",
                                                    cache_dir=config.MODEL_DIR)
            _pipe.model.generation_config = base
        except Exception as exc:
            logger.warning("could not attach base whisper generation_config: %s", exc)
        logger.info("Fine-tuned Whisper loaded.")
    return _pipe


def transcribe_words(wav_path: str, lang_code: str | None = "te") -> dict:
    """Whole-file decode with word timestamps. Returns the same shape as
    whisper_service.transcribe_words: {"words":[{start,end,word,prob}], "text",
    "language"}. Word probabilities are unavailable from the HF pipeline -> prob
    is None (slice_words treats None probs as unscored). Never raises."""
    empty = {"words": [], "text": "", "language": lang_code}
    try:
        pipe = _load()
        out = pipe(
            wav_path,
            return_timestamps="word",
            generate_kwargs={"task": "transcribe", "language": lang_code or "te"},
        )
        words = []
        for ch in (out.get("chunks") or []):
            token = (ch.get("text") or "").strip()
            ts = ch.get("timestamp") or (None, None)
            if not token or ts[0] is None or ts[1] is None:
                continue
            words.append({"start": round(float(ts[0]), 3), "end": round(float(ts[1]), 3),
                          "word": token, "prob": None})
        return {"words": words, "text": (out.get("text") or "").strip(),
                "language": lang_code}
    except Exception as exc:
        logger.warning("Fine-tuned Whisper failed for %s (lang=%s): %s",
                       wav_path, lang_code, exc)
        return empty
