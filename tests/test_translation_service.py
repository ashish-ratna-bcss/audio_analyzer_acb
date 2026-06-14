import pytest
from unittest.mock import MagicMock, patch


def test_translate_returns_string():
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_inputs = MagicMock()
    mock_tokenizer.return_value = mock_inputs
    mock_inputs.to.return_value = mock_inputs

    mock_output = MagicMock()
    mock_model.generate.return_value = mock_output
    mock_tokenizer.batch_decode.return_value = ["Hello sir"]

    with patch("services.translation_service._tokenizer", mock_tokenizer), \
         patch("services.translation_service._model", mock_model):
        from services.translation_service import translate
        result = translate("సార్ నమస్కారం", src_lang="tel_Telu", tgt_lang="eng_Latn")

    assert isinstance(result, str)
    assert result == "Hello sir"


def test_translate_segments():
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_inputs = MagicMock()
    mock_tokenizer.return_value = mock_inputs
    mock_inputs.to.return_value = mock_inputs
    mock_tokenizer.batch_decode.return_value = ["translated text"]
    mock_model.generate.return_value = MagicMock()

    segments = [
        {"speaker": "Speaker_1", "start": 0.0, "end": 2.0, "text": "నమస్కారం"},
    ]

    with patch("services.translation_service._tokenizer", mock_tokenizer), \
         patch("services.translation_service._model", mock_model):
        from services.translation_service import translate_segments
        result = translate_segments(segments, src_lang="tel_Telu", tgt_lang="eng_Latn")

    assert result[0]["translated_text"] == "translated text"
    assert result[0]["text"] == "నమస్కారం"
