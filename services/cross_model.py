def selfcheck_confidence(enh_text: str, org_text: str, *, embed_fn) -> dict:
    """Single-model validation: agreement between IndicConformer on the enhanced
    clip vs the original clip. Replaces 3-way cross-model consensus now that there
    is one ASR model. Agreement = embedding cosine penalized by length-ratio
    divergence. Returns {agreement, confidence} in [0,1]; never raises.

    Both runs are the SAME model, so high agreement means enhancement did not
    change the words (stable, trustworthy); low agreement means the enhanced and
    original decodes diverge -> flag for review.
    """
    a, b = (enh_text or "").strip(), (org_text or "").strip()
    if not a or not b:
        return {"agreement": 0.0, "confidence": 0.0}
    try:
        cos = float(embed_fn(a, b))
    except Exception:
        return {"agreement": 0.0, "confidence": 0.0}
    la, lb = len(a.split()), len(b.split())
    length_ratio = min(la, lb) / max(la, lb) if max(la, lb) else 0.0
    agreement = round(max(0.0, min(1.0, cos)) * length_ratio, 3)
    return {"agreement": agreement, "confidence": agreement}


def normalized_edit_distance(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 1.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] / max(la, lb)


def _pairwise_agreement(texts, embed_fn):
    """Mean pairwise cosine over the given (key,text) pairs, plus per-key mean sim."""
    keys = list(texts)
    if len(keys) < 2:
        return (1.0 if keys else 0.0), {k: 1.0 for k in keys}
    sims = {k: [] for k in keys}
    total, n = 0.0, 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            s = float(embed_fn(texts[keys[i]], texts[keys[j]]))
            sims[keys[i]].append(s)
            sims[keys[j]].append(s)
            total += s
            n += 1
    mean = total / n if n else 0.0
    per_key = {k: (sum(v) / len(v) if v else 0.0) for k, v in sims.items()}
    return mean, per_key


def compare_passes(texts, confidences, *, embed_fn, agreement_min=0.6, conf_thresh=0.5):
    """Validate the 3 ASR passes together. Script-agnostic (embeddings).

    Returns consensus pick + agreement + flag. Never raises: embedding failure
    degrades to agreement 0.0 + flagged.
    """
    nonempty = {k: v.strip() for k, v in texts.items() if v and v.strip()}

    # Confidence mean over real (non-None) scores only.
    real_confs = [c for c in confidences.values() if isinstance(c, (int, float))]
    mean_conf = round(sum(real_confs) / len(real_confs), 3) if real_confs else 0.0

    if len(nonempty) < 2:
        only = next(iter(nonempty), None)
        return {"consensus_pass": only,
                "consensus_text": nonempty.get(only, "") if only else "",
                "agreement": 0.0, "confidence": mean_conf,
                "flagged": True, "flag_reason": "insufficient_passes"}

    try:
        agreement, per_key = _pairwise_agreement(nonempty, embed_fn)
        embed_ok = True
    except Exception:
        agreement, per_key, embed_ok = 0.0, {k: 0.0 for k in nonempty}, False

    # Consensus = medoid (highest mean similarity to peers); tie-break by confidence.
    def _score(k):
        return (per_key.get(k, 0.0), confidences.get(k) or 0.0)

    consensus_pass = max(nonempty, key=_score)
    consensus_text = nonempty[consensus_pass]

    if not embed_ok or agreement < agreement_min:
        return {"consensus_pass": consensus_pass, "consensus_text": consensus_text,
                "agreement": round(agreement, 3), "confidence": mean_conf,
                "flagged": True, "flag_reason": "cross_model_disagreement"}
    if mean_conf < conf_thresh:
        return {"consensus_pass": consensus_pass, "consensus_text": consensus_text,
                "agreement": round(agreement, 3), "confidence": mean_conf,
                "flagged": True, "flag_reason": "low_confidence"}
    return {"consensus_pass": consensus_pass, "consensus_text": consensus_text,
            "agreement": round(agreement, 3), "confidence": mean_conf,
            "flagged": False, "flag_reason": None}
