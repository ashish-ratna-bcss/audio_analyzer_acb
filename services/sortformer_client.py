"""Thin HTTP client to the Sortformer sidecar (used by the BASE image).

Stdlib-only (urllib) so the base image gains no new dependency. The sidecar
reads the audio by path from the shared case_data volume, so we send only the
path, never the bytes.
"""
import json
import logging
import os
import urllib.request

import config

logger = logging.getLogger(__name__)

SORTFORMER_URL = os.getenv("SORTFORMER_URL", "http://sortformer:9000")
SORTFORMER_TIMEOUT = float(os.getenv("SORTFORMER_TIMEOUT", "600"))


def diarize_with_overlap(audio_path: str, num_speakers: int | None = None) -> list[dict]:
    """Call the sidecar /diarize. Returns [{start,end,speaker}]. Raises on failure
    so the caller can fall back to pyannote."""
    payload = json.dumps({"audio_path": audio_path, "num_speakers": num_speakers}).encode()
    req = urllib.request.Request(
        f"{SORTFORMER_URL}/diarize", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=SORTFORMER_TIMEOUT) as resp:
        data = json.loads(resp.read().decode())
    return data.get("segments", [])
