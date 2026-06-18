def test_wrappers_import_and_expose_callables():
    from services import vad_service, enhancement_service, separation_service
    assert callable(vad_service.detect_speech)
    assert callable(enhancement_service.enhance)
    assert callable(separation_service.separate_vocals)
