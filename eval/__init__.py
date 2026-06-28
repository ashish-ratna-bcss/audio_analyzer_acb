"""Offline ASR evaluation harness — measures pipeline output against verified
transcripts (WER / CER / Script-Fidelity-Rate / number / entity accuracy) with a
per-engine breakdown. Fully air-gapped; the foundation for the verbatim-ASR
program (see docs/superpowers/specs/2026-06-28-verbatim-asr-program-design.md)."""
