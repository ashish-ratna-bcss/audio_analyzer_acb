"""Release forensic model VRAM after a job (L7 — idle VRAM management).

The ASR/audio models are lazily loaded on the first use of each job and cached
in module globals. Keeping them resident holds ~10+ GB of VRAM even while
forensic is idle, starving the co-located OCR service. `unload_all()` (called in
run_pipeline's finally) drops every cached model and empties the CUDA cache, so
an idle forensic worker frees the GPU; the next job lazily reloads. Cost: a
one-time reload (~30-60s) at the start of each job — acceptable for batch
forensic, and the price of sharing one GPU between two services.
"""
import gc
import importlib
import logging

import config

logger = logging.getLogger(__name__)

# module -> cache globals to reset (must match each service's lazy-load globals).
_CACHES = {
    "services.whisper_service": ["_model"],
    "services.telugu_asr_service": ["_pipe"],
    "services.indic_asr_service": ["_model"],
    "services.lang_id_service": ["_processor", "_model"],
    "services.diarization_service": ["_pipeline"],
    "services.embedding_service": ["_model"],
    "services.vad_service": ["_model"],
    "services.separation_service": ["_model", "_sepformer"],
    "services.enhancement_service": ["_state"],
}


def unload_all() -> list:
    """Null every cached model global + empty the CUDA cache. Returns the names
    freed (for logging/tests). No-op when ASR_UNLOAD_AFTER_JOB is false. Never
    raises — VRAM release must not fail a completed job."""
    if not getattr(config, "ASR_UNLOAD_AFTER_JOB", True):
        return []
    freed = []
    for mod_name, attrs in _CACHES.items():
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for attr in attrs:
            if getattr(mod, attr, None) is not None:
                try:
                    setattr(mod, attr, None)
                    freed.append(f"{mod_name.split('.')[-1]}.{attr}")
                except Exception:
                    pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    if freed:
        logger.info("released model VRAM (%d): %s", len(freed), ", ".join(freed))
    return freed
