# Forensic Transcription Notes

This service transcribes sensitive evidence audio (e.g. ACB trap-case
recordings). Accuracy, completeness and defensibility take priority over
readability. These notes record the decisions that protect evidentiary value.

## ASR output is an aid, not authoritative

Automatic transcription of code-switched Telugu/English phone audio is **not
100% accurate**. Output WILL contain misrecognized words. Before any
evidentiary use, a human must verify the transcript against the source audio.
Treat the JSON as an indexed first pass, not a certified transcript.

- Preserve the original audio unmodified (chain of custody).
- Keep the per-segment `confidence`, `no_speech_prob`, `compression_ratio` —
  low confidence marks spans needing closer human review.
- Record the model + config used (below) alongside any exported transcript.

## Why there is NO `initial_prompt`

An `initial_prompt` *primes* Whisper toward expected vocabulary. On evidence
audio that causes **word substitution** — the model emits prompt words instead
of what was actually said — and an English prompt forced English output even
for Telugu speech. Both destroy verbatim fidelity. No prompt = faithful to the
spoken language and words. Do not re-add a prompt.

## Decode configuration (services/whisper_service.py)

| setting | value | reason |
|---|---|---|
| model | large-v3 | most accurate Whisper |
| beam_size | 10 | recovered the cost/amount exchange beam_size=5 dropped |
| condition_on_previous_text | False | True snowballed a hallucinated phrase into a repetition loop, losing most speech |
| initial_prompt | none | priming causes word substitution (see above) |
| vad_filter | True (min_silence 700ms, pad 400ms) | VAD OFF lost more content in testing; conservative VAD does not clip speech here |
| temperature | default fallback | escapes failed/looping windows |
| word_timestamps | True | word-level timing for evidence review |

## No silent dropping (api/routes/stt.py)

Every transcribed segment is returned. The system never deletes a segment for
"low confidence" or "repetition" — that could discard real speech and is not
defensible. Suspect spans are flagged via their metrics for human review.

## Verification history

Tuned against three real call recordings (charan-RTA, audio_call,
Deepika-ashish) by comparing full output text to the known conversation,
selecting the config with the most complete + faithful transcript.
