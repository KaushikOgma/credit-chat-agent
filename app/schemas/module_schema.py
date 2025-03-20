from typing import List
from pydantic import BaseModel
from fastapi import UploadFile

class QAGenerateRequest(BaseModel):
    text: str


class QAEvaluationSchema(BaseModel):
    question: str
    answer: str

class QAEvaluationRequest(BaseModel):
    data: List[QAEvaluationSchema]


class FileUploadRequest(BaseModel):
    files: List[UploadFile]