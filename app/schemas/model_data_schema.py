from enum import Enum
from pydantic import BaseModel, Field, UUID4
from typing import List, Optional
from datetime import datetime


class ModelDataSortFields(str, Enum):
    createdAt_ASC = "createdAt:ASC"
    createdAt_DESC = "createdAt:DESC"
    updatedAt_ASC = "updatedAt:ASC"
    updatedAt_DESC = "updatedAt:DESC"



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


