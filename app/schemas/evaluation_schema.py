from enum import Enum
from pydantic import BaseModel, Field, UUID4
from typing import List, Optional
from datetime import datetime


class EvalDataSortFields(str, Enum):
    createdAt_ASC = "createdAt:ASC"
    createdAt_DESC = "createdAt:DESC"
    updatedAt_ASC = "updatedAt:ASC"
    updatedAt_DESC = "updatedAt:DESC"



class SaveEvalQASchema(BaseModel):
    question: str
    answer: str
    isActive: Optional[bool] = True


class UpdateEvalQASchema(BaseModel):
    question: str
    answer: str
    isActive: Optional[bool] = True


class EvalQASchema(BaseModel):
    question: str
    answer: str
    fileName: Optional[str] = None
    isActive: Optional[bool] = True
    createdAt:  Optional[datetime] = None
    updatedAt:  Optional[datetime] = None


class EvalQARequestSchema(BaseModel):
    data: List[EvalQASchema]


class EvalQAResponseSchema(BaseModel):
    data: List[EvalQASchema] = None
    message: str

