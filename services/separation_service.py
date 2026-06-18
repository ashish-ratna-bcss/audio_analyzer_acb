import config

_model = None


def load_demucs():
    global _model
    if _model is None:
        from demucs.pretrained import get_model
        _model = get_model(config.DEMUCS_MODEL)
    return _model


def separate_vocals(in_wav: str, out_wav: str) -> str:
    """HTDemucs vocal-stem isolation. Advisory input to VAD only — gated, opt-in."""
    import torchaudio
    from demucs.apply import apply_model
    model = load_demucs()
    wav, sr = torchaudio.load(in_wav)
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    sources = apply_model(model, wav[None], device="cpu")[0]
    vocals = sources[model.sources.index("vocals")]
    torchaudio.save(out_wav, vocals, model.samplerate)
    return out_wav
