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


def _max_pairwise_distance(texts):
    keys = list(texts)
    worst = 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = normalized_edit_distance(texts[keys[i]], texts[keys[j]])
            worst = max(worst, d)
    return worst


def compare_passes(texts, confidences, vad_positive, embedding_sim,
                   *, edit_thresh=0.4, sim_thresh=0.6, conf_thresh=0.5) -> dict:
    nonempty = {k: v for k, v in texts.items() if v and v.strip()}

    # 1. VAD said speech but a pass returned nothing -> lowest confidence.
    if vad_positive and len(nonempty) < len(texts):
        return {"confidence": 0.0, "disagreement": True, "flagged": True,
                "flag_reason": "vad_positive_asr_empty"}

    mean_conf = round(sum(confidences.values()) / max(len(confidences), 1), 3)
    worst = _max_pairwise_distance(nonempty) if len(nonempty) > 1 else 0.0
    disagreement = worst > edit_thresh and (embedding_sim is None or embedding_sim < sim_thresh)

    # 2. Cross-model disagreement.
    if disagreement:
        return {"confidence": mean_conf, "disagreement": True, "flagged": True,
                "flag_reason": "cross_model_disagreement"}
    # 3. Low confidence.
    if mean_conf < conf_thresh:
        return {"confidence": mean_conf, "disagreement": False, "flagged": True,
                "flag_reason": "low_logprob_confidence"}
    return {"confidence": mean_conf, "disagreement": False, "flagged": False,
            "flag_reason": None}
