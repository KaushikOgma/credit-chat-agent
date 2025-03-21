
from datetime import timezone, timedelta
import datetime
import os
from app.db import get_db
from fastapi import Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse, Response
from fastapi import HTTPException
from app.schemas.auth_schema import credential_exception, invalid_credential_resp
from typing import Optional
from pymongo.database import Database
from app.utils.config import settings
from app.utils.helpers.date_helper import get_user_time
from app.utils.logger import setup_logger

logger = setup_logger()

# Define the API key dependency
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_current_user(db: Database = Depends(get_db), api_key: Optional[str] = Depends(api_key_header), response: Response = None):
    """**Summary:**
    validate the acces token and return the current user for the access token

    **Args:**
        - db (Database): db session referance
        - api_key (String): api key
        - response (Response): fat api Response object
    """
    try:
        if api_key is None:
            raise credential_exception
        else:
            user_data = db.user.find_one({
                "apiKey": api_key
            })
            if not user_data:
                raise credential_exception
            else:
                return user_data
    except Exception as error:
        logger.exception(error)
        raise error
