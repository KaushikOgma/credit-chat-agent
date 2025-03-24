from fastapi import APIRouter, File, Request, Depends, status, Query
from typing import List, Union
from pymongo.database import Database
from app.controllers.finetune_controller import FinetuneController
from fastapi.responses import JSONResponse
from starlette import status as starlette_status
from app.schemas.finetune_schema import (
    FinetuneDataSortFields, 
    TrainQARequestSchema, 
    TrainQAResponseSchema, 
    TrainQASchema,
    SaveTrainQASchema,
    UpdateTrainQASchema
)
from app.dependencies.finetune_dependencies import get_finetune_controller
from fastapi.exceptions import HTTPException
from app.db import get_db
import datetime
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()
router = APIRouter()


@router.get("/get_train_data", response_model=TrainQAResponseSchema)
async def get_train_data(
    fileName: str = Query(None, description="fileName"),
    isActive: bool = Query(None, description="isActive"),
    startDate: str =  Query(None, description=f"startDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    endDate: str =  Query(None, description=f"endDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    sortBy: List[FinetuneDataSortFields] = Query([FinetuneDataSortFields.createdAt_DESC], description=f"sortBy"),
    finetune_controller: FinetuneController = Depends(get_finetune_controller),
    db_instance: Database = Depends(get_db)
):
    
    try:
        sort_params = []
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
            data = await finetune_controller.get_train_data(db, startDate, endDate, fileName, isActive, sort_params)
            return JSONResponse(
                        status_code=200, content={"data": data, "message": "Data fetched successfully"}
                    )
    except Exception as error:
        logger.exception(error)
        return JSONResponse(content={"message": str(error)}, status_code=500)



@router.post("/add_train_data", status_code=status.HTTP_201_CREATED)
async def add_metadata(
    body: List[SaveTrainQASchema],
    finetune_controller: FinetuneController = Depends(get_finetune_controller),
    db_instance: Database = Depends(get_db)
):
    
    try:
        # Convert the request body to a dictionary
        data = [elm.model_dump() for elm in body]

        # Call the add_user method of the user controller
        async with db_instance as db:
            data = await finetune_controller.add_train_data(data, db)
            return JSONResponse(
                status_code=200, content={"data": data, "message": "Data added successfully"}
            )
    except Exception as error:
        # Log the error
        logger.exception(error)
        # Return a JSON response with an error message
        return JSONResponse(content={"message": str(error)}, status_code=500)



@router.post("/update_train_data/{id}", status_code=status.HTTP_201_CREATED)
async def update_train_data(
    id: str,
    body: UpdateTrainQASchema,
    finetune_controller: FinetuneController = Depends(get_finetune_controller),
    db_instance: Database = Depends(get_db)
):
    try:
        # Convert the request body to a dictionary
        data = body.model_dump(exclude_unset=True)

        # Call the add_user method of the user controller
        async with db_instance as db:
            update_flag = await finetune_controller.update_train_data(id, data, db)
            if update_flag:
                return JSONResponse(
                            status_code=200, content={"message": "Data updated successfully"}
                        )
            else:
                return JSONResponse(
                            status_code=400, content={"message": "Invalid request"}
                        )
    except Exception as error:
        # Log the error
        logger.exception(error)
        # Return a JSON response with an error message
        return JSONResponse(content={"message": str(error)}, status_code=500)



@router.delete("/delete_train_data", response_model=TrainQAResponseSchema)
async def delete_train_data(
    fileName: str = Query(None, description="fileName"),
    isActive: bool = Query(None, description="isActive"),
    startDate: str =  Query(None, description=f"startDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    endDate: str =  Query(None, description=f"endDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    finetune_controller: FinetuneController = Depends(get_finetune_controller),
    db_instance: Database = Depends(get_db)
):
    
    try:
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
        async with db_instance as db:
            await finetune_controller.delete_train_data(db, startDate, endDate, fileName, isActive)
            return JSONResponse(
                status_code=200, content={"message": "Data deleted successfully"}
            )
    except Exception as error:
        logger.exception(error)
        return JSONResponse(content={"message": str(error)}, status_code=500)

