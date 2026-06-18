def test_imports():
    from services import indic_asr_service, embedding_service, diarization_service
    assert callable(indic_asr_service.transcribe_clip)
    assert callable(embedding_service.similarity)
    assert callable(diarization_service.diarize_with_overlap)
