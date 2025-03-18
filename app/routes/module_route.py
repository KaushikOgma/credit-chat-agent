from fastapi import APIRouter, Request, Depends, status, Query
from typing import Union
from app.controllers import qa_controller
from app.controllers import data_ingestion_controller
from app.controllers import qa_evaluation_controller
from app.schemas.module_schema import QAGenerateRequest, QAEvaluationRequest
from app.schemas.module_schema import FolderPathRequest
from langchain.schema import Document as LCDocument
router = APIRouter()

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
    return await qa_controller.generate_qa(req_data)

# @router.post("/extract", status_code=status.HTTP_200_OK)
# async def extract_text_from_folder(
#     request: Request,
#     req_data: FolderPathRequest
# ):
#     """
#     **Summary:**
#     Extract text from files in a specified folder.

#     This endpoint processes the files in the provided folder path and extracts text from supported file types.

#     **Args:**
#     - request (Request): The incoming request object.
#     - req_data (FolderPathRequest): The folder path request schema.
    
#     **Returns:**
#     A list of extracted texts with metadata.
#     """
#     input_folder = req_data.folder_path
#     if not input_folder or not os.path.isdir(input_folder):
#         raise HTTPException(status_code=400, detail="Invalid folder path")

#     documents = process_folder(input_folder)
#     return [{"filename": doc.metadata["filename"], "content": doc.page_content} for doc in documents]



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
    return await qa_evaluation_controller.get_evaluation_score(req_data.data)
