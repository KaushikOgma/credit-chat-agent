from typing import List
from pydantic import BaseModel
from fastapi import UploadFile


class FileUploadRequest(BaseModel):
    files: List[UploadFile]