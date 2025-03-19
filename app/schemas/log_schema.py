from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class LogType(str, Enum):
    ERROR = "ERROR"
    INFO = "INFO"
    WARNING = "WARNING"


class statusType(str, Enum):
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"

class LogEntry(BaseModel):
    message: str
    type: Optional[LogType] = LogType.INFO.value
    stackTrace: Optional[str] = None
    timestamp:  Optional[datetime] = None
    

class LogSortFields(str, Enum):
    createdAt_ASC = "createdAt:ASC"
    createdAt_DESC = "createdAt:DESC"
    updatedAt_ASC = "updatedAt:ASC"
    updatedAt_DESC = "updatedAt:DESC"



class LogSchema(BaseModel):
    id: Optional[int] = Field(None, alias="_id")
    message: str
    status: Optional[statusType] = statusType.ERROR.value
    logTrail: Optional[List[LogEntry]] = []
    moduleName: Optional[str] = None
    serviceName: Optional[str] = None
    createdAt:  Optional[datetime] = None
    updatedAt:  Optional[datetime] = None
    

class SaveLogSchema(BaseModel):
    id: Optional[int] = Field(None, alias="_id")
    message: str
    type: LogType = LogType.INFO
    stackTrace: Optional[str] = None
    moduleName: Optional[str]
    serviceName: Optional[str] = None


class LogSchemasResponse(BaseModel):
    data: List[LogSchema] = None
    message: str
