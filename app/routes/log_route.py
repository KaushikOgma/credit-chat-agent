from fastapi import APIRouter, Request, Depends, status, Query
from typing import List, Union
from pymongo.database import Database
from app.controllers import log_controller
from fastapi.responses import JSONResponse
from starlette import status as starlette_status
from app.schemas.log_schema import LogSchemasResponse, LogSortFields, SaveLogSchema
from fastapi.exceptions import HTTPException
from app.db import get_db
import datetime
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()
router = APIRouter()

@router.get("/get_logs", response_model=LogSchemasResponse)
async def get_logs(
    moduleName: str = Query(None, description="moduleName"),
    serviceName: str = Query(None, description="serviceName"),
    type: str = Query(None, description="log type"),
    startDate: str =  Query(None, description=f"startDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    endDate: str =  Query(None, description=f"endDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdA"),
    sortBy: List[LogSortFields] = Query([LogSortFields.createdAt_DESC], description=f"sortBy"),
    db: Database = Depends(get_db)
):
    """**Summary:**
    fetch all logs.

    **Args:**
    - `moduleName` (str): The name of the module for which logs should be fetched.
    - `serviceName` (str): The name of the service for which logs should be fetched.
    - `startDate` (str): The start date of the logs to be fetched in mentioned format. If not provided, all logs from the beginning of time are fetched.
    - `endDate` (str): The end date of the logs to be fetched in mentioned format. If not provided, all logs until the end of time are fetched.
    - `sortBy` (List[LogSortFields]): The fields to sort the logs by. For example, ["createdAt:ASC"] or ["createdAt:DESC"].
    - `db` (Database): Dependency to get the database session.

    **Returns:**
        - `LogSchemasResponse`: List of logs.
    """
    sort_params = []
    try:
        # Validate startDate
        if startDate is not None:
            try:
                if "+" not in startDate:
                    startDate = startDate.replace(" ","+")
                # Validate the date format
                startDate = datetime.datetime.strptime(startDate, settings.ACCEPTED_DATE_TIME_STRING)
                startDate = startDate.replace(hour=0, minute=0, second=0)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"startDate must be in {settings.ACCEPTED_DATE_TIME_STRING} format")

        # Validate endDate
        if endDate is not None:
            try:
                if "+" not in endDate:
                    endDate = endDate.replace(" ","+")
                # Validate the date format
                endDate = datetime.datetime.strptime(endDate, settings.ACCEPTED_DATE_TIME_STRING)
                endDate = endDate.replace(hour=23, minute=59, second=59)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"endDate must be in {settings.ACCEPTED_DATE_TIME_STRING} format")

        # Validate sort parameters
        if sortBy is not None and len(sortBy) > 0:
            for item in sortBy:
                [field, order] = item.value.split(":")
                if field not in [field for (field, _) in sort_params]:
                    sort_params.append((field, 1 if order == "ASC" else -1))
            # Convert the list of tuples into a dictionary
            sort_params = {field: order for field, order in sort_params}

        # Call the controller to fetch the logs
        return await log_controller.get_logs(db, moduleName, serviceName, startDate, endDate, type, sort_params)
    except Exception as error:
        logger.exception(error)
        return JSONResponse(content={"message": str(error)}, status_code=500)


@router.post("/add_log", status_code=starlette_status.HTTP_200_OK)
async def add_log(
    body: SaveLogSchema,
    db: Database = Depends(get_db)
):
    """**Summary:**
    This method is responsible for adding new log.

    **Args:**
    - `body` (SaveRequestSchema): request body for adding new request.
    - `db` (Database): Dependency to get the database session.

    """
    try:
        data = body.model_dump()
        return await log_controller.add_log(data, db)
    except Exception as error:
        logger.exception(error)
        return JSONResponse(content={"message": str(error)}, status_code=500)

