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

class CreditReportExtractor:    
    def __init__(self):
        self.app_key = settings.ARRAY_APP_KEY
        self.server_token = settings.ARRAY_SERVER_TOKEN
        self.array_url = settings.ARRAY_BASE_URL
        self.regenerate_token_url = "authenticate/v2/usertoken"
        self.order_credit_report_url = "report/v2"
        self.fetch_credit_report_url = "report/v2"
        self.productCode = "exp1bReportScore"
        self.ttlInMinutes = "60"
        self.service_name = "credit_report_service"
 
    async def regenerate_token(self, user_id: str):
        """Regenerates the user token asynchronously."""
        try:
            url = f"{self.array_url}/{self.regenerate_token_url}"
            payload = {"ttlInMinutes": self.ttlInMinutes, "userId": user_id, "appKey": self.app_key}
            headers = {
                "accept": "application/json",
                "x-array-server-token": self.server_token,
                "content-type": "application/json",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"success": True, "message": "Token regenerated successfully", "data": data}
                    elif response.status == 400:
                        return {"success": False, "message": "Bad Request - userId or reportKey is missing or invalid"}
                    elif response.status == 403:
                        return {"success": False, "message": "Forbidden - Invalid or missing authorization header"}
                    elif response.status == 404:
                        return {"success": False, "message": "Not Found - Report is no longer accessible (expired or deleted)"}
                    else:
                        return {"success": False, "message": f"Unexpected Error {response.status}"}
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return None
 

    async def order_credit_report(self, user_id: str, user_token: str):
        """Orders the credit report asynchronously."""
        try:
            url = f"{self.array_url}/{self.order_credit_report_url}"
            payload = {"productCode": self.productCode, "userId": user_id}
            headers = {
                "accept": "application/json",
                "x-array-server-token": self.server_token,
                "x-array-user-token": user_token,
                "content-type": "application/json",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"success": True, "message": "Token regenerated successfully.", "data": data}
                    elif response.status == 400:
                        return {"success": False, "message": "Invalid or missing userId or productCode."}
                    elif response.status == 401:
                        return {"success": False, "message": "Customer's identity is not verified."}
                    elif response.status == 403:
                        return {"success": False, "message": "Invalid or missing API authentication token."}
                    elif response.status == 404:
                        return {"success": False, "message": "Resource not found."}
                    elif response.status == 406:
                        return {"success": False, "message": "Permission denied for the requested product."}
                    elif response.status == 408:
                        return {"success": False, "message": "Request was cancelled by the customer."}
                    elif response.status == 422:
                        return {"success": False, "message": "Missing essential identifying information."}
                    else:
                        return {"success": False, "message": f"Unexpected error (status {response.status})."}
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return None
 

    async def retrieve_credit_report(self, report_key: str, display_token: str):
        """Retrieves the credit report from Array asynchronously."""
        try:
            url = f"{self.array_url}/{self.fetch_credit_report_url}?reportKey={report_key}&displayToken={display_token}&live=false"
            headers = {"content-type": "application/json"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "message": "Operation successful. Report delivered.",
                            "data": data,
                        }
                    elif response.status == 202:
                        return {
                            "success": False,
                            "message": "Request accepted, but report is still being generated. Try again later.",
                        }
                    elif response.status == 204:
                        return {
                            "success": False,
                            "message": "Report could not be generated. Check bureau error headers.",
                        }
                    elif response.status == 400:
                        return {"success": False, "message": "Invalid or missing reportKey query parameter."}
                    elif response.status == 401:
                        return {"success": False, "message": "DisplayToken expired. Regenerate the token."}
                    elif response.status == 404:
                        return {"success": False, "message": "ReportKey expired (older than 30 days) or not found."}
                    elif response.status == 408:
                        return {"success": False, "message": "Customer cancelled the request."}
                    elif response.status == 412:
                        return {"success": False, "message": "Invalid content-type header value."}
                    else:
                        return {"success": False, "message": f"Unexpected status {response.status}."}
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return None
 

    async def get_credit_report(self, user_id: str):
        """Combines all steps to get the final credit report asynchronously."""
        try:
            user_token_resp = await self.regenerate_token(user_id)
            if not user_token_resp["success"]:
                raise Exception(user_token_resp["message"])
            user_token_data = user_token_resp["data"]
            user_token = user_token_data.get("userToken")
            if not user_token:
                raise Exception("Missing user token in response.")
    
            order_data_response = await self.order_credit_report(user_token)
            if not order_data_response["success"]:
                raise Exception(order_data_response["message"])
            order_data = order_data_response["data"]
            report_key = order_data.get("reportKey")
            display_token = order_data.get("displayToken")
            if not report_key or not display_token:
                raise Exception("Missing reportKey or displayToken in order response.")
    
            report_response = await self.retrieve_credit_report(report_key, display_token)
            if not report_response["success"]:
                raise Exception(report_response["message"])
            report = report_response["data"]

            # credit_report_json_path = os.path.join(".", settings.LOCAL_UPLOAD_LOCATION ,"array_data.json")
            # report = {}
            # with open(credit_report_json_path, "r") as f:
            #     report = json.load(f)
            return report
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return None
 
