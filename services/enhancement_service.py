import config

_state = None


def load_dfn():
    global _state
    if _state is None:
        from df.enhance import init_df
        model, df_state, _ = init_df()
        _state = (model, df_state)
    return _state


def enhance(in_wav: str, out_wav: str) -> str:
    """DeepFilterNet3 denoise. Parallel branch — never replaces the original."""
    import torch
    from df.enhance import enhance as df_enhance, load_audio, save_audio
    model, df_state = load_dfn()
    audio, _ = load_audio(in_wav, sr=df_state.sr())
    # Disable cuDNN during DF inference — its Conv layers trigger
    # CUDNN_STATUS_NOT_SUPPORTED on A10G which permanently corrupts the CUDA
    # context for all subsequent GPU calls (Whisper, MMS-LID, pyannote, etc.)
    prev = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        enhanced = df_enhance(model, df_state, audio)
    finally:
        torch.backends.cudnn.enabled = prev
    save_audio(out_wav, enhanced, df_state.sr())
    return out_wav
