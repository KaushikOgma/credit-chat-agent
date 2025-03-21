"""
This Module is responsible for Managing
all of the token actions

Usage:
    from app.utils.helpers.auth_helper import generate_api_key

    apiKey = generate_api_key(prefix="CGAI-", length=26)
"""

from datetime import datetime, timedelta, timezone
import string
import secrets
from fastapi import HTTPException
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()


def generate_api_key(prefix="CGAI-", length=26):
    """**Summary:**
    Generate an API key with a given prefix and length.

    **Args:**
    - `prefix` (str, optional): The prefix to prepend to the generated key. Defaults to "CGAI-".
    - `length` (int, optional): The length of the randomly generated part of the key. Defaults to 26.

    Returns:
    - `api_key` (str): The generated API key.

    """
    try:
        # Generate the random part
        random_part = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))
        
        # Prepend the prefix
        api_key = prefix + random_part

        # Return the API key
        return api_key
    except Exception as error:
        logger.exception(error)
        raise error
    
# Export the required function
__all__ = ["generate_api_key"]
