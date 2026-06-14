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
