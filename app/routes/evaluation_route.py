from fastapi import APIRouter, File, Request, Depends, status, Query
from typing import List, Union
from pymongo.database import Database
from app.controllers.evaluation_controller import EvaluationController
from fastapi.responses import JSONResponse
from starlette import status as starlette_status
from app.schemas.evaluation_schema import (
    EvalQARequestSchema, 
    EvalQAResponseSchema, 
    EvalDataSortFields, 
    EvalQASchema,
    SaveEvalQASchema,
    UpdateEvalQASchema
)
from app.dependencies.evaluation_dependencies import get_eval_controller
from fastapi.exceptions import HTTPException
from app.db import get_db
import datetime
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()
router = APIRouter()


@router.get("/get_eval_data", response_model=EvalQAResponseSchema)
async def get_eval_data(
    fileName: str = Query(None, description="fileName"),
    isActive: bool = Query(None, description="isActive"),
    isProcessed: bool = Query(None, description="isProcessed"),
    startDate: str =  Query(None, description=f"startDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    endDate: str =  Query(None, description=f"endDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    sortBy: List[EvalDataSortFields] = Query([EvalDataSortFields.createdAt_DESC], description=f"sortBy"),
    eval_controller: EvaluationController = Depends(get_eval_controller),
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
            data = await eval_controller.get_eval_data(db, startDate, endDate, fileName, isActive, isProcessed, sort_params)
            return JSONResponse(
                        status_code=200, content={"data": data, "message": "Data fetched successfully"}
                    )
    except Exception as error:
        logger.exception(error)
        return JSONResponse(content={"message": str(error)}, status_code=500)



@router.post("/add_eval_data", status_code=status.HTTP_201_CREATED)
async def add_eval_data(
    body: List[SaveEvalQASchema],
    eval_controller: EvaluationController = Depends(get_eval_controller),
    db_instance: Database = Depends(get_db)
):
    
    try:
        # Convert the request body to a dictionary
        data = [
            {
                **elm.model_dump(), 
                "fileName": elm.fileName if hasattr(elm, "fileName") else None, 
                "fileType": elm.fileType if hasattr(elm, "fileType") else None, 
            } 
            for elm in body
        ]

        # Call the add_user method of the user controller
        async with db_instance as db:
            data = await eval_controller.add_eval_data(data, db)
            return JSONResponse(
                        status_code=200, content={"data": data, "message": "Data added successfully"}
                    )
    except Exception as error:
        # Log the error
        logger.exception(error)
        # Return a JSON response with an error message
        return JSONResponse(content={"message": str(error)}, status_code=500)



@router.post("/update_eval_data/{id}", status_code=status.HTTP_201_CREATED)
async def update_eval_data(
    id: str,
    body: UpdateEvalQASchema,
    eval_controller: EvaluationController = Depends(get_eval_controller),
    db_instance: Database = Depends(get_db)
):
    try:
        # Convert the request body to a dictionary
        data = body.model_dump(exclude_unset=True)
        if "fileName" not in data:
            data["fileName"] = None
        if "fileType" not in data:
            data["fileType"] = None

        # Call the add_user method of the user controller
        async with db_instance as db:
            update_flag = await eval_controller.update_eval_data(id, data, db)
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




@router.delete("/delete_eval_data", response_model=EvalQAResponseSchema)
async def delete_eval_data(
    fileName: str = Query(None, description="fileName"),
    isActive: bool = Query(None, description="isActive"),
    isProcessed: bool = Query(None, description="isProcessed"),
    startDate: str =  Query(None, description=f"startDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    endDate: str =  Query(None, description=f"endDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    eval_controller: EvaluationController = Depends(get_eval_controller),
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
            await eval_controller.delete_eval_data(db, startDate, endDate, fileName, isProcessed, isActive)
            return JSONResponse(
                        status_code=200, content={"message": "Data deleted successfully"}
                    )
    except Exception as error:
        logger.exception(error)
        return JSONResponse(content={"message": str(error)}, status_code=500)




@router.post("/initiate_evaluating", status_code=status.HTTP_200_OK)
async def initiate_evaluating(
    fileName: str = Query(None, description="fileName of the eval data file"),
    isActive: bool = Query(None, description="isActive"),
    startDate: str =  Query(None, description=f"startDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    endDate: str =  Query(None, description=f"endDate in {settings.ACCEPTED_DATE_TIME_STRING} format to filter createdAt"),
    modelId: str =  Query(..., description="Model Id which we need to evaluate"),
    eval_controller: EvaluationController = Depends(get_eval_controller),
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
        await eval_controller.initiate_evaluating(startDate, endDate, fileName, isActive, modelId)
        return JSONResponse(
            status_code=200, content={"message": "Data deleted successfully"}
        )
    except Exception as error:
        logger.exception(error)
        return JSONResponse(content={"message": str(error)}, status_code=500)

