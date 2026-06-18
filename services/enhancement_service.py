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
    from df.enhance import enhance as df_enhance, load_audio, save_audio
    model, df_state = load_dfn()
    audio, _ = load_audio(in_wav, sr=df_state.sr())
    enhanced = df_enhance(model, df_state, audio)
    save_audio(out_wav, enhanced, df_state.sr())
    return out_wav
