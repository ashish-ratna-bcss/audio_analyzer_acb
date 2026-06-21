"""Fire-and-forget job status webhooks.

If a job was submitted with a `callback_url`, the pipeline POSTs a small JSON
payload to it on every stage transition and terminal state — the integration's
substitute for polling. Stdlib-only (urllib) so the base image gains no new
dependency. Failures NEVER break the pipeline: a webhook is best-effort.
"""
import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT = 5.0


def notify(callback_url: str, payload: dict) -> None:
    """POST payload as JSON to callback_url. Swallows all errors."""
    if not callback_url:
        return
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            callback_url, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT).close()
    except Exception as e:  # noqa: BLE001 — webhook must never break the job
        logger.warning("webhook POST to %s failed: %s", callback_url, e)
