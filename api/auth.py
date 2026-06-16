from fastapi import Header, HTTPException
from typing import Optional

import config


async def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Reject requests without a valid X-API-Key header.

    If config.API_KEY is empty (e.g. local dev), auth is disabled and all
    requests pass. In deployment set API_KEY in the environment/.env.
    """
    if not config.API_KEY:
        return
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
