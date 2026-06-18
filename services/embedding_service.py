import config

_model = None


def load_embed():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBED_MODEL, token=config.PYANNOTE_AUTH_TOKEN or None)
    return _model


def similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    from sentence_transformers import util
    model = load_embed()
    emb = model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(emb[0], emb[1]).item())
