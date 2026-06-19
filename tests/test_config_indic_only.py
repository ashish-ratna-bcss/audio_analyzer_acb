import config


def test_indic_only_config_defaults():
    assert config.INDIC_SELFCHECK_MIN == 0.6
    assert config.INDIC_CONF_MIN == 0.5
