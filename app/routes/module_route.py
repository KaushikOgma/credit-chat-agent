from fastapi import APIRouter, Request, Depends, status, Query
from typing import Union
from app.controllers import qa_controller
from app.schemas.module_schema import QAGenerateRequest
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