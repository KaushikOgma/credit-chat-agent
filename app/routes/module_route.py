from fastapi import APIRouter, Request, Depends, status, Query, File
from typing import Union

from fastapi.responses import JSONResponse
from app.controllers import qa_controller
from app.controllers import data_ingestion_controller
from app.schemas.module_schema import FileUploadRequest
from app.controllers import qa_evaluation_controller
from app.schemas.module_schema import QAGenerateRequest, QAEvaluationRequest
from langchain.schema import Document as LCDocument
from app.utils.logger import setup_logger
from app.utils.config import settings

router = APIRouter()
logger = setup_logger()

@router.post("/generate_qa", status_code=status.HTTP_200_OK)
async def get_modules(
    request: Request,
    req_data: QAGenerateRequest
):
    
    """**Summary:**
    Fetches a paginated list of modules along with their submodules.

    This method retrieves a subset of modules from the database based on the specified
    page number and page size.

    **Args:**
    - db: The database session object.
    - page (Union[None,int]): The page number to retrieve.
    - page_size (Union[None,int]): The number of items per page.
    - verified_token (Boolean): It will indicate if the request is authenticated or not.
    """
    serviceName = "qa_generator"
    try:
        return await qa_controller.generate_qa(req_data)
    except Exception as error:
        logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": serviceName})
        return JSONResponse(content={"message": str(error)}, status_code=500)


@router.post("/evaluate", status_code=status.HTTP_200_OK)
async def evaluate_system_accuracy(
    request: Request,
    req_data: QAEvaluationRequest
):
    """
    **Summary:**
    Evaluates the accuracy of a question answering system.

    This endpoint evaluates the accuracy of a question answering system by comparing the system's responses
    to the actual answers.

    **Args:**
    - request (Request): The incoming request object.
    - req_data (QAEvaluationRequest): The QA evaluation request schema.

    **Returns:**
    The evaluation score of the question answering system.
    """
    serviceName = "qa_evaluator"
    try:
        return await qa_evaluation_controller.get_evaluation_score(req_data.data)
    except Exception as error:
        logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": serviceName})
        return JSONResponse(content={"message": str(error)}, status_code=500)

