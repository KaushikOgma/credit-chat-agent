from fastapi import APIRouter, File, Request, Depends, status, Query
from typing import List, Union
from pymongo.database import Database
from app.controllers.data_ingestion_controller import DataIngestionController
from fastapi.responses import JSONResponse
from starlette import status as starlette_status
from app.schemas.data_ingestion_schema import FileUploadRequest
from app.dependencies.data_ingestion_dependencies import get_data_ingestion_controller
from fastapi.exceptions import HTTPException
from app.db import get_db
import datetime
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()
router = APIRouter()



@router.post("/import_train_data", status_code=status.HTTP_201_CREATED)
async def add_metadata(
    req_data: FileUploadRequest = File(...),
    data_ingestion_controller: DataIngestionController = Depends(get_data_ingestion_controller),
    db_instance: Database = Depends(get_db)
):
    
    try:
        async with db_instance as db:
            return await data_ingestion_controller.import_training_data(db, req_data)
    except Exception as error:
        # Log the error
        logger.exception(error)
        # Return a JSON response with an error message
        return JSONResponse(content={"message": str(error)}, status_code=500)




@router.post("/import_evaluation_data", status_code=status.HTTP_201_CREATED)
async def add_metadata(
    req_data: FileUploadRequest = File(...),
    data_ingestion_controller: DataIngestionController = Depends(get_data_ingestion_controller),
    db_instance: Database = Depends(get_db)
):
    
    try:
        async with db_instance as db:
            return await data_ingestion_controller.import_evaluation_data(db, req_data)
    except Exception as error:
        # Log the error
        logger.exception(error)
        # Return a JSON response with an error message
        return JSONResponse(content={"message": str(error)}, status_code=500)

