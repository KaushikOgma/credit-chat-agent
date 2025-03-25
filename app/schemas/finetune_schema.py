from enum import Enum
from pydantic import BaseModel, Field, UUID4
from typing import List, Optional
from datetime import datetime


class FinetuneDataSortFields(str, Enum):
    createdAt_ASC = "createdAt:ASC"
    createdAt_DESC = "createdAt:DESC"
    updatedAt_ASC = "updatedAt:ASC"
    updatedAt_DESC = "updatedAt:DESC"



class SaveTrainQASchema(BaseModel):
    question: str
    answer: str
    isActive: Optional[bool] = True


class UpdateTrainQASchema(BaseModel):
    question: str
    answer: str
    isActive: Optional[bool] = True


class ModelDataSchema(BaseModel):
    id: Optional[UUID4] = Field(None, alias="_id")
    dataset_ids: Optional[List[str]] = None
    file_id: Optional[str] = None
    job_id: Optional[str] = None
    model_id: Optional[str] = None
    params: Optional[dict] = None
    metrices: Optional[dict] = None
    createdAt:  Optional[datetime] = None
    updatedAt:  Optional[datetime] = None


class TrainQASchema(BaseModel):
    question: str
    answer: str
    fileName: Optional[str] = None
    metadataId: Optional[str] = None
    isActive: Optional[bool] = True
    createdAt:  Optional[datetime] = None
    updatedAt:  Optional[datetime] = None


class TrainQARequestSchema(BaseModel):
    data: List[TrainQASchema]


class TrainQAResponseSchema(BaseModel):
    data: List[TrainQASchema] = None
    message: str

