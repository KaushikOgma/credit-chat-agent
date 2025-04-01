from fastapi import APIRouter, File, Request, Depends, status, Query
from typing import List, Union
from pymongo.database import Database
from app.controllers.credit_report_controller import CreditReportController
from fastapi.responses import JSONResponse
from starlette import status as starlette_status
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.dependencies.chat_report_dependencies import get_credit_report_controller
from fastapi.exceptions import HTTPException
from app.db import get_db
import datetime
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()
router = APIRouter()



@router.post("/fetch_context", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def test_chat(
    user_id: str = Query(..., description="user_id"),
    user_query: str = Query(..., description="user_id"),
    credit_report_controller: CreditReportController = Depends(get_credit_report_controller),
    db_instance: Database = Depends(get_db)
):
    
    try:
        async with db_instance as db:
            response = await credit_report_controller.get_credit_report_context(db, user_id, user_query)
            return JSONResponse(
                        status_code=200, content={"data": response, "message": "Got Context successfully"}
                    )
    except Exception as error:
        # Log the error
        logger.exception(error)
        # Return a JSON response with an error message
        return JSONResponse(content={"message": str(error)}, status_code=500)


