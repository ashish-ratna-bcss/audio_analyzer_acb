"""
MMS-LID-256 audio-grounded language identification.
Runs BEFORE any ASR pass — independent of Whisper's decoder.
Outputs ISO 639-3 codes; use to_iso639_1() to get 2-letter routing codes.
"""
import logging
import config

logger = logging.getLogger(__name__)

_processor = None
_model = None

# MMS-LID ISO 639-3 → Whisper/routing ISO 639-1
_MMS_TO_ISO1: dict[str, str] = {
    "tel": "te", "hin": "hi", "tam": "ta", "kan": "kn", "mal": "ml",
    "ben": "bn", "guj": "gu", "pan": "pa", "mar": "mr", "urd": "ur",
    "asm": "as", "npi": "ne", "ori": "or", "sin": "si", "san": "sa",
    "eng": "en", "arb": "ar", "fra": "fr", "deu": "de", "spa": "es",
    "jpn": "ja", "kor": "ko", "cmn": "zh", "zho": "zh", "rus": "ru",
    "ita": "it", "por": "pt", "nld": "nl", "pol": "pl", "tur": "tr",
    "vie": "vi", "tha": "th", "ind": "id", "dan": "da", "fin": "fi",
    "swe": "sv", "nor": "no", "nob": "no", "snd": "sd", "pus": "ps",
    "fas": "fa", "swh": "sw", "hau": "ha", "yor": "yo", "ibo": "ig",
    "mya": "my", "khm": "km", "lao": "lo", "kat": "ka", "hye": "hy",
    "azj": "az", "kaz": "kk", "uzb": "uz", "tat": "tt", "mon": "mn",
    "srp": "sr", "hrv": "hr", "bos": "bs", "slv": "sl", "ces": "cs",
    "slk": "sk", "pol": "pl", "ukr": "uk", "bel": "be", "bul": "bg",
    "ron": "ro", "hun": "hu", "ell": "el", "heb": "he", "lit": "lt",
    "lav": "lv", "est": "et", "isl": "is", "mlt": "mt", "glg": "gl",
    "cat": "ca", "eus": "eu",
}


def to_iso639_1(mms_code: str | None) -> str | None:
    """Convert MMS-LID 3-letter code to ISO 639-1 2-letter code."""
    if not mms_code:
        return None
    return _MMS_TO_ISO1.get(mms_code.lower())


def vote_file_language(per_clip_lids, *, allowed_langs, min_conf):
    """Majority-vote a file-level language prior (ISO 639-1) from per-clip MMS-LID.

    Only clips with top1_confidence >= min_conf count. If allowed_langs is
    non-empty, only languages in it are eligible. Returns None if no eligible vote.
    """
    from collections import Counter

    counts = Counter()
    for lid in per_clip_lids:
        if (lid.get("top1_confidence") or 0.0) < min_conf:
            continue
        iso1 = to_iso639_1(lid.get("top1"))
        if not iso1:
            continue
        if allowed_langs and iso1 not in allowed_langs:
            continue
        counts[iso1] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _load():
    global _processor, _model
    if _model is None:
        from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
        import torch
        logger.info("Loading MMS-LID-256 (%s)…", config.MMS_LID_MODEL)
        _processor = AutoFeatureExtractor.from_pretrained(config.MMS_LID_MODEL)
        _model = Wav2Vec2ForSequenceClassification.from_pretrained(config.MMS_LID_MODEL)
        if config.WHISPER_DEVICE == "cuda":
            _model = _model.to("cuda")
        _model.eval()
        logger.info("MMS-LID-256 loaded.")
    return _processor, _model


def identify(wav_path: str) -> dict:
    """
    Returns top-2 language IDs from raw audio, independent of Whisper.
    Result keys: top1 (ISO 639-3), top2, top1_confidence, top2_confidence.
    Returns all-None dict on failure (caller falls back to Whisper lang).
    """
    import torch
    import librosa

    _empty = {
        "top1": None, "top2": None,
        "top1_confidence": 0.0, "top2_confidence": 0.0,
        "candidates": [],
    }

    try:
        processor, model = _load()
        waveform, _ = librosa.load(wav_path, sr=16000, mono=True)
        if len(waveform) < 800:  # <50ms — too short for reliable LID
            return _empty

        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        device = "cuda" if config.WHISPER_DEVICE == "cuda" else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        top2 = torch.topk(probs, k=2)
        id2label = model.config.id2label

        candidates = [
            {"language": id2label[idx.item()], "confidence": round(score.item(), 4)}
            for score, idx in zip(top2.values, top2.indices)
        ]

        return {
            "top1": candidates[0]["language"],
            "top2": candidates[1]["language"] if len(candidates) > 1 else None,
            "top1_confidence": candidates[0]["confidence"],
            "top2_confidence": candidates[1]["confidence"] if len(candidates) > 1 else 0.0,
            "candidates": candidates,
        }

    except Exception as exc:
        logger.warning("MMS-LID failed for %s: %s", wav_path, exc)
        return _empty
