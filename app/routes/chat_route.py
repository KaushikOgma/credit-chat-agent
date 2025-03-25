from fastapi import APIRouter, File, Request, Depends, status, Query
from typing import List, Union
from pymongo.database import Database
from app.controllers.chat_controller import ChatController
from fastapi.responses import JSONResponse
from starlette import status as starlette_status
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.dependencies.chat_dependencies import get_chat_controller
from fastapi.exceptions import HTTPException
from app.db import get_db
import datetime
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()
router = APIRouter()



@router.post("/test", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def test_chat(
    req_data: ChatRequest,
    model_data_id: str = Query(..., description="Model Id with which you want to test the chat"),
    chat_controller: ChatController = Depends(get_chat_controller),
    db_instance: Database = Depends(get_db)
):
    
    try:
        async with db_instance as db:
            response = await chat_controller.test_chat(db, req_data, model_data_id)
            return JSONResponse(
                        status_code=200, content={"data": response, "message": "Got Answer successfully"}
                    )
    except Exception as error:
        # Log the error
        logger.exception(error)
        # Return a JSON response with an error message
        return JSONResponse(content={"message": str(error)}, status_code=500)


