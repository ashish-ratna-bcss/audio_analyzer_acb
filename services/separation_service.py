import os
import logging

import config

logger = logging.getLogger(__name__)

_model = None
_sepformer = None


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


def load_sepformer():
    """SpeechBrain SepFormer speaker-separation model (lazy, GPU if available)."""
    global _sepformer
    if _sepformer is None:
        from speechbrain.inference.separation import SepformerSeparation
        run_opts = {}
        if config.WHISPER_DEVICE == "cuda":
            run_opts["device"] = "cuda"
        logger.info("Loading SepFormer (%s)…", config.SEPFORMER_MODEL)
        _sepformer = SepformerSeparation.from_hparams(
            source=config.SEPFORMER_MODEL,
            savedir=os.path.join(config.MODEL_DIR, "sepformer"),
            run_opts=run_opts,
        )
        logger.info("SepFormer loaded.")
    return _sepformer


def separate_speakers(in_wav: str, out_dir: str, prefix: str) -> list[str]:
    """
    Split a cross-talk clip into per-speaker streams using SepFormer.

    Returns a list of 16kHz mono wav paths (one per separated source). Each is
    transcribed independently downstream so both/all overlapped voices are
    recovered. Returns [] on failure so the caller can fall back to the mixed
    clip rather than dropping the segment.
    """
    try:
        import torch
        import torchaudio

        model = load_sepformer()
        # est_sources: (batch, time, n_src), at the model's sample rate (8 kHz).
        est = model.separate_file(path=in_wav)
        if est.ndim == 3:
            est = est[0]                      # (time, n_src)
        n_src = est.shape[-1]
        model_sr = int(getattr(model.hparams, "sample_rate", 8000))

        paths = []
        for i in range(n_src):
            src = est[:, i].detach().cpu().unsqueeze(0)   # (1, time)
            # Normalize to avoid clipping on save, then resample to 16k for ASR.
            peak = src.abs().max()
            if peak > 0:
                src = src / peak * 0.95
            if model_sr != 16000:
                src = torchaudio.functional.resample(src, model_sr, 16000)
            out = os.path.join(out_dir, f"{prefix}_spk{i}.wav")
            torchaudio.save(out, src, 16000)
            paths.append(out)
        return paths
    except Exception as exc:
        logger.warning("SepFormer separation failed for %s: %s", in_wav, exc)
        return []
