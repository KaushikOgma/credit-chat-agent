from enum import Enum
from pydantic import BaseModel, Field, UUID4
from typing import List, Optional
from datetime import datetime


class FinetuneDataSortFields(str, Enum):
    createdAt_ASC = "createdAt:ASC"
    createdAt_DESC = "createdAt:DESC"
    updatedAt_ASC = "updatedAt:ASC"
    updatedAt_DESC = "updatedAt:DESC"



class TrainQASchema(BaseModel):
    question: str
    answer: str
    fileName: Optional[str] = None
    isProcessed: Optional[bool] = False
    isActive: Optional[bool] = True
    createdAt:  Optional[datetime] = None
    updatedAt:  Optional[datetime] = None


class TrainQARequestSchema(BaseModel):
    data: List[TrainQASchema]


class TrainQAResponseSchema(BaseModel):
    data: List[TrainQASchema] = None
    message: str

