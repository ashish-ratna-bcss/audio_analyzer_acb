import config


def test_multimodel_config_defaults():
    assert config.ALLOWED_LANGS == set()           # empty = open auto
    assert config.LID_VOTE_MIN_CONF == 0.5
    assert config.NO_SPEECH_MAX == 0.6
    assert config.AGREEMENT_MIN == 0.6
    assert config.LOUDNORM_LUFS == -16.0
    assert config.GAP_WINDOW_S == 10.0
    assert isinstance(config.GHOST_PHRASES, list)
    assert "thank you" in [p.lower() for p in config.GHOST_PHRASES]
