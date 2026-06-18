import config


def test_recall_config_present():
    assert config.VAD_THRESHOLD == 0.25
    assert config.VAD_MIN_SPEECH_MS == 100
    assert config.VAD_SPEECH_PAD_MS_L3 == 300
    assert config.VAD_MIN_SILENCE_MS_L3 == 100
    assert config.DFN_MODEL == "DeepFilterNet3"
    assert config.DEMUCS_MODEL == "htdemucs_ft"
