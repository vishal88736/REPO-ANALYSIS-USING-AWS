from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
from app.config import settings

API_KEY_HEADER = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.api_secret_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key
