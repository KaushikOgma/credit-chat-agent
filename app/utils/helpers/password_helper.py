"""
This Module is responsible for Managing
all of the common actions

Usage:
    from app.utils.helpers.password_helper import verify_password

    verify_password(plain_password, hashed_password)
"""

import bcrypt
from app.utils.logger import setup_logger
logger = setup_logger()


async def hash_password(plain_password: str) -> str:
    """
    Hash a plain password using bcrypt.

    Args:
        plain_password (str): The plain password to hash.

    Returns:
        str: The hashed password.
    """
    try:
        # Generate a salt and hash the password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
        return hashed_password.decode("utf-8")
    except Exception as error:
        logger.exception("hash_password:: error - " + str(error))
        raise error


async def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.

    Args:
        plain_password (str): The plain password to verify.
        hashed_password (str): The hashed password to verify against.

    Returns:
        bool: True if the password matches, False otherwise.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except Exception as error:
        logger.exception("verify_password:: error - " + str(error))
        raise error


# Export the required function
__all__ = ["hash_password", "verify_password"]
