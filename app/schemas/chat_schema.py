from typing import List, Optional
from pydantic import BaseModel
from fastapi import UploadFile


class ChatAgentRequest(BaseModel):
    user_query: str
    user_id: str
    is_verified: Optional[bool] = False
    is_premium: Optional[bool] = False

class ChatRequest(BaseModel):
    question: str

class ChatSchema(BaseModel):
    question: str
    answer: str

class ChatResponse(BaseModel):
    data: ChatSchema
    message: str