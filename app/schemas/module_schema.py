from typing import List
from pydantic import BaseModel

class QAGenerateRequest(BaseModel):
    text: str


class QAEvaluationSchema(BaseModel):
    question: str
    answer: str

class QAEvaluationRequest(BaseModel):
    data: List[QAEvaluationSchema]


class FolderPathRequest(BaseModel):
    folder_path: str