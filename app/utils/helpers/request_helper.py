import asyncio
import json
import re
import aiohttp
import requests
import traceback
from typing import List, Union
import sys
import os
sys.path.append(os.getcwd())
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()

TEST_ARRAY_USER_ID = settings.TEST_ARRAY_USER_ID
TEST_USER_ID = settings.TEST_USER_ID
SERVER_URL = settings.SERVER_URL
ASK_CREDIT_GENIUS_ROUT = "chat/askCreditGenius"


async def start_chat(query):
    """Retrieves the credit report from Array asynchronously."""
    try:
        url = f"{SERVER_URL}/{ASK_CREDIT_GENIUS_ROUT}"
        payload = {
            "user_input": query,
            "userId": TEST_USER_ID,
            "creditServiceUserId": TEST_ARRAY_USER_ID,
            "is_premium": True,
            "is_verified": True
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    error_message = data.get("error_message", None)
                    data["error_message"] = error_message
                    if error_message is not None:
                        # data["response"] = 
                        return {
                            "success": False,
                            "message": "Operation Not Successful",
                            "data": data,
                        }
                    else:
                        return {
                            "success": True,
                            "message": "Operation successful. Report delivered.",
                            "data": data,
                        }
                else:
                    return {"success": False, "message": f"Unexpected status {response.status}."}
    except Exception as error:
        logger.exception(error, extra={"moduleName": settings.MODULE})
        return None
