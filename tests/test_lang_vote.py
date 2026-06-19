from services.lang_id_service import vote_file_language


def test_majority_vote():
    lids = [
        {"top1": "tel", "top1_confidence": 0.9},
        {"top1": "tel", "top1_confidence": 0.8},
        {"top1": "eng", "top1_confidence": 0.7},
    ]
    assert vote_file_language(lids, allowed_langs=set(), min_conf=0.5) == "te"


def test_low_conf_excluded():
    lids = [
        {"top1": "kor", "top1_confidence": 0.2},
        {"top1": "tel", "top1_confidence": 0.9},
    ]
    assert vote_file_language(lids, allowed_langs=set(), min_conf=0.5) == "te"


def test_allowed_set_filters():
    lids = [
        {"top1": "kor", "top1_confidence": 0.9},
        {"top1": "kor", "top1_confidence": 0.9},
        {"top1": "tel", "top1_confidence": 0.8},
    ]
    # kor not in allowed -> te wins despite fewer votes
    assert vote_file_language(lids, allowed_langs={"te", "en", "hi"}, min_conf=0.5) == "te"


def test_all_low_conf_returns_none():
    lids = [{"top1": "tel", "top1_confidence": 0.1}]
    assert vote_file_language(lids, allowed_langs=set(), min_conf=0.5) is None


def test_empty_returns_none():
    assert vote_file_language([], allowed_langs=set(), min_conf=0.5) is None
