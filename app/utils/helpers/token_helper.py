"""
This Module is responsible for Managing
all of the token actions

Usage:
    from app.utils.helpers.token_helper import create_access_token

    token = create_access_token({"user_id": 1})
"""

from datetime import datetime, timedelta, timezone
from jose import jwt
from fastapi import HTTPException
from app.utils.config import settings
from app.utils.logger import LogHandler

logger = LogHandler.get_logger()


def create_access_token(token_data: dict, expires_delta: timedelta = None):
    """**Summary:**
    Create JWT access token keping the token_data as the payload of the token

    **Args:**
        - token_data (Dict): access token payload that needts to be imposed in the token
        - expires_delta (timedelta): access token expiration time
    """
    try:
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=int(settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            )
        token_data.update({"exp": expire})
        jwt_token = jwt.encode(token_data, settings.ENCRYPTION_KEY, algorithm="HS256")
        return jwt_token
    except Exception as error:
        logger.exception("create_access_token:: error - " + str(error))
        return None


def create_refresh_token(token_data: dict, expires_delta: timedelta = None):
    """**Summary:**
    Create JWT refresh token keping the token_data as the payload of the token

    **Args:**
        - token_data (Dict): refresh token payload that needts to be imposed in the token
        - expires_delta (timedelta): refresh token expiration time
    """
    try:
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                days=int(settings.REFRESH_TOKEN_EXPIRE_DAYS)
            )
        token_data.update({"exp": expire})
        jwt_token = jwt.encode(token_data, settings.ENCRYPTION_KEY, algorithm="HS256")
        return jwt_token
    except Exception as error:
        logger.exception("create_refresh_token:: error - " + str(error))
        return None


def decode_access_token(token: str):
    """
    Decode and validate a JWT access token.

    Args:
        token (str): The JWT token to decode.

    Returns:
        Dict: The decoded token data.

    Raises:
        HTTPException: If the token is invalid or expired.
    """
    try:
        decoded_token = jwt.decode(token, settings.ENCRYPTION_KEY, algorithms=["HS256"])
        return decoded_token
    except jwt.ExpiredSignatureError as error:
        logger.exception("verify_token:: error - " + str(error))
        raise HTTPException(status_code=401, detail="Token has expired") from error
    except Exception as error1:
        logger.exception("verify_token:: error1 - " + str(error1))
        raise HTTPException(status_code=401, detail="Invalid token") from error1


# Export the required function
__all__ = ["create_access_token", "create_refresh_token", "decode_access_token"]
