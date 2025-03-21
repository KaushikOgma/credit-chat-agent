from fastapi import APIRouter, Request, Depends, status, Query
from typing import List, Union
from pymongo.database import Database
from app.controllers.metadata_controller import MetadataController
from fastapi.responses import JSONResponse
from starlette import status as starlette_status
from app.schemas.metadata_schema import (
    MetadataSortFields,
    MetadataSchema, 
    MetadataDetailSchemasResponse, 
    MetadataSchemasResponse,
    SaveMetadataSchema,
    UpdateMetadataSchema
)
from app.dependencies.metadata_dependencies import get_metadata_controller
from fastapi.exceptions import HTTPException
from app.db import get_db
import datetime
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()
router = APIRouter()

@router.get("/get_metadatas", response_model=MetadataSchemasResponse)
async def get_metadatas(
    fileName: str = Query(None, description="fileName"),
    isTrainData: bool = Query(None, description="isTrainData"),
    isProcessed: bool = Query(None, description="isProcessed"),
    startDate: str =  Query(None, description=f"startDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    endDate: str =  Query(None, description=f"endDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    sortBy: List[MetadataSortFields] = Query([MetadataSortFields.createdAt_DESC], description=f"sortBy"),
    metadata_controller: MetadataController = Depends(get_metadata_controller),
    db_instance: Database = Depends(get_db)
):
    
    try:
        sort_params = []
        # if not current_user["isAdmin"]:
        #     userId = str(current_user["_id"])
        if startDate is not None:
            try:
                if "+" not in startDate:
                    startDate = startDate.replace(" ","+")
                # Validate the date format
                startDate = datetime.datetime.strptime(startDate, settings.ACCEPTED_DATE_TIME_STRING)
                startDate = startDate.replace(hour=0, minute=0, second=0)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"startDate must be in {settings.ACCEPTED_DATE_TIME_STRING} format")
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
        async with db_instance as db:
            return await metadata_controller.get_metadatas(db, startDate, endDate, fileName, isTrainData, isProcessed, sort_params)
    except Exception as error:
        logger.exception(error)
        return JSONResponse(content={"message": str(error)}, status_code=500)



@router.get("/get_metadata_details/{id}", response_model=MetadataDetailSchemasResponse)
async def get_metadata_details(
    id: str,
    metadata_controller: MetadataController = Depends(get_metadata_controller),
    db_instance: Database = Depends(get_db)
):
    """Fetch the current user's details.

    **Args:**
    - `current_user` (User): The current user details of the logged in user.
    - `db` (Database): Dependency to get the database session.

    **Returns:**
    - `UserDetails`: The current user's details.
    """
    try:
        # Fetch the current user's details from the database
        async with db_instance as db:
            return await metadata_controller.get_metadata_detail(db, id)
    except Exception as error:
        # Log the error and return a JSON response with the error message
        logger.exception(error)
        return JSONResponse(content={"message": str(error)}, status_code=500)



@router.post("/add_metadata", status_code=status.HTTP_201_CREATED)
async def add_metadata(
    body: SaveMetadataSchema,
    metadata_controller: MetadataController = Depends(get_metadata_controller),
    db_instance: Database = Depends(get_db)
):
    
    try:
        # Convert the request body to a dictionary
        data = body.model_dump()

        # Call the add_user method of the user controller
        async with db_instance as db:
            return await metadata_controller.add_metadata(data, db)
    except Exception as error:
        # Log the error
        logger.exception(error)
        # Return a JSON response with an error message
        return JSONResponse(content={"message": str(error)}, status_code=500)



@router.post("/update_metadata/{id}", status_code=status.HTTP_201_CREATED)
async def update_metadata(
    id: str,
    body: UpdateMetadataSchema,
    metadata_controller: MetadataController = Depends(get_metadata_controller),
    db_instance: Database = Depends(get_db)
):
    try:
        # Convert the request body to a dictionary
        data = body.model_dump(exclude_unset=True)

        # Call the add_user method of the user controller
        async with db_instance as db:
            return await metadata_controller.update_metadata(id, data, db)
    except Exception as error:
        # Log the error
        logger.exception(error)
        # Return a JSON response with an error message
        return JSONResponse(content={"message": str(error)}, status_code=500)
