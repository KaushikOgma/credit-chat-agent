from enum import Enum
from pydantic import BaseModel, Field, UUID4
from typing import List, Optional
from datetime import datetime


class MetadataSortFields(str, Enum):
    createdAt_ASC = "createdAt:ASC"
    createdAt_DESC = "createdAt:DESC"
    updatedAt_ASC = "updatedAt:ASC"
    updatedAt_DESC = "updatedAt:DESC"



class SaveMetadataSchema(BaseModel):
    content: str
    isTrainData: bool = True


class UpdateMetadataSchema(BaseModel):
    content: Optional[str] = None
    isTrainData: Optional[bool] = True



class MetadataSchema(BaseModel):
    id: Optional[UUID4] = Field(None, alias="_id")
    fileName: Optional[str] = None
    fileType: Optional[str] = None
    content: Optional[str] = None
    isTrainData: Optional[bool] = True
    isProcessed: Optional[bool] = False
    createdAt:  Optional[datetime] = None
    updatedAt:  Optional[datetime] = None



class MetadataSchemasResponse(BaseModel):
    data: List[MetadataSchema] = None
    message: str

class MetadataDetailSchemasResponse(BaseModel):
    data: MetadataSchema
    message: str