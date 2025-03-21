from typing import List
import datetime
from fastapi.responses import JSONResponse
from pymongo.database import Database
from app.utils.helpers.common_helper import generate_uuid
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.attribute_selector.log_attributes import LogProjections
from app.repositories.log_repositories import LogRepository
from app.utils.config import settings
from app.utils.constants import DBCollections
from app.utils.logger import setup_logger
logger = setup_logger()

class LogController:

    def __init__(self, log_repo: LogRepository):
        self.log_repo = log_repo
        self.service_name = "logging_service"

    async def get_logs(self,
                        db: Database, 
                        moduleName: str = None, 
                        serviceName: str = None, 
                        userId: str = None, 
                        startDate: datetime = None, 
                        endDate: datetime = None, 
                        type: str = None, 
                        sort_params: list = None) -> dict:
        """**Summary:**
        This method is responsible for fetching all logs based on the given input parameters.

        **Args:**
        - `db` (Database): db session referance.
        - `moduleName` (str, optional): Module Name. Defaults to None.
        - `serviceName` (str, optional): Service Name. Defaults to None.
        - `startDate` (datetime, optional): Start date for the time range. Defaults to None.
        - `endDate` (datetime, optional): End date for the time range. Defaults to None.
        - `type` (str, optional): Log type. Defaults to None.
        - `sort_params` (list, optional): Sort parameters. Defaults to None.

        **Returns:**
            - A dictionary containing the log data and a success message or an error message.
        """
        try:
            filterData = {}
            input_timezone = None
            if moduleName is not None:
                filterData["moduleName"] = moduleName
            if serviceName is not None:
                filterData["serviceName"] = serviceName
            if userId is not None:
                filterData["userId"] = userId
            if type is not None:
                filterData["type"] = type
            if startDate is not None:
                input_timezone = startDate.tzname().replace("UTC","")
                filterData["createdAt"] = {
                    '$gte': convert_timezone(startDate, to_string=False, timeZone="UTC"),
                }
                if endDate is None:
                    # If end date is not provided, set the end date to the start date of the next day
                    endDate = startDate.replace(hour=23, minute=59, second=59)
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
                else:
                    # If end date is provided, set the end date in the query to the given end date
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
            # Fetch orders from db
            data = await self.log_repo.get_log(db, filterData, sort_params, input_timezone)
            return JSONResponse(
                status_code=200, content={"data": data, "message": "Data fetched successfully"}
            )
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return JSONResponse(
                status_code=500,
                content={
                    "message": "get_logs:: error - " + str(error),
                },
            )

    async def add_log(self, data, db):
        """**Summary:**
        This method is responsible for logging/adding log.

        **Args:**
        - `data` (dict): request body for adding new request.
        - db (Database): db session reference
        """
        try:
            log_id = await self.log_repo.add_log(db, data)
            return JSONResponse(
                status_code=200, content={"id": log_id,"message": "Data inserted successfully"}
            )
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error