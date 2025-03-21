from enum import Enum
from pydantic import BaseModel, Field, UUID4
from typing import List, Optional
from datetime import datetime


class UserSortFields(str, Enum):
    createdAt_ASC = "createdAt:ASC"
    createdAt_DESC = "createdAt:DESC"
    updatedAt_ASC = "updatedAt:ASC"
    updatedAt_DESC = "updatedAt:DESC"


class UserSchema(BaseModel):
    id: Optional[UUID4] = Field(None, alias="_id")
    name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    contactAddress: Optional[str] = None
    contactNo: Optional[str] = None
    isActive: Optional[bool] = True
    isAdmin: Optional[bool] = False
    config: Optional[dict] = {}
    userId: Optional[str] = None
    apiKey: Optional[str] = None
    createdAt:  Optional[datetime] = None
    updatedAt:  Optional[datetime] = None
    

class SaveUserSchema(BaseModel):
    id: Optional[UUID4] = Field(None, alias="_id")
    name: str
    email: str
    password: str
    contactAddress: Optional[str]
    contactNo: Optional[str]
    isActive: Optional[bool] = True
    isAdmin: Optional[bool] = False
    userId: Optional[str] = None
    config: Optional[dict] = {}
    apiKey: Optional[str] = None


class UpdateUserSchema(BaseModel):
    id: Optional[int] = Field(None, alias="_id")
    name: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    contactAddress: Optional[str] = None
    contactNo: Optional[str] = None
    isActive: Optional[str] = None
    isAdmin: Optional[str] = None
    userId: Optional[str] = None
    config: Optional[dict] = None
    apiKey: Optional[str] = None


class UserSchemasResponse(BaseModel):
    data: List[UserSchema] = None
    message: str

class UserDetailSchemasResponse(BaseModel):
    data: UserSchema
    message: str