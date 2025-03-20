from typing import List
import datetime
from fastapi.responses import JSONResponse
from app.schemas.log_schema import LogEntry
from pymongo.database import Database
from app.utils.helpers.common_helper import generate_uuid
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.attribute_selector.log_attributes import LogProjections
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()

async def get_logs(db: Database, 
                    moduleName: str = None, 
                    serviceName: str = None, 
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
        # Perform the aggregation pipeline
        pipeline = [
            {"$match": filterData},
            # project the fields which need to be included in the final result
            {"$project": LogProjections.get_all_attribute(input_timezone)},
            # sort the fields depending on sort params
            {"$sort": sort_params}
        ]
        data = list(db.log.aggregate(pipeline))
        return {"data": data, "message": "Data fetched successfully"}
    except Exception as error:
        logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": serviceName})
        return JSONResponse(
            status_code=500,
            content={
                "message": "get_logs:: error - " + str(error),
            },
        )

async def add_log(data, db):
    """**Summary:**
    This method is responsible for logging/adding log.

    **Args:**
    - `data` (dict): request body for adding new request.
    - db (Database): db session reference
    """
    serviceName = None
    try:
        if "id" in data:
            del data["id"]
        newLogId = generate_uuid()
        data["_id"] = newLogId
        data["createdAt"] = get_user_time()
        data["updatedAt"] = get_user_time()
        log_entry = {
            "message": data["message"],
            "type": data.get("type"),
            "stackTrace": data.get("stackTrace"),
            "timestamp": get_user_time()
        }
        log_filter = { }
        if "serviceName" in data and data["serviceName"] is not None:
            serviceName = data["serviceName"]
            log_filter["serviceName"] = data["serviceName"]
        if len(log_filter) > 0:
            log_filter["moduleName"] = data["moduleName"]

        existing_log = db.log.find_one(log_filter)
        if len(log_filter) == 0 or not existing_log:
            data["logTrail"] = [log_entry]
            if data["type"].value == "INFO":
                data["status"] = "SUCCESS"
            else:
                data["status"] = "ERROR"
            del data["type"]
            del data["stackTrace"]
            db.log.insert_one(data)
        else:
            existing_log["logTrail"].append(log_entry)
            if data["type"].value == "INFO":
                existing_log["status"] = "SUCCESS"
            else:
                existing_log["status"] = "ERROR"
            existing_log["message"] = data["message"]
            existing_log["updatedAt"] = get_user_time()
            del data["type"]
            del data["stackTrace"]
            db.log.update_one(
                {"_id": existing_log["_id"]},
                {"$set": existing_log}
            )
        return {"message": "Data inserted successfully"}
    except Exception as error:
        logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": serviceName})
        raise error