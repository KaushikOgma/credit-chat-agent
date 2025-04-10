from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from fastapi import UploadFile


class ChatAgentRequest(BaseModel):
    user_query: str
    user_id: str
    credit_service_user_id: Optional[str] = None
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


class ChatHistoryRequestSchema(BaseModel):
    user_id: str
    credit_service_user_id: Optional[str] = None
    before_id: Optional[str] = None
    size: Optional[int] = None


class CreditGeniusResponseSchema(BaseModel):
    question : str 
    response : str 
    traversed_path : str
    premium_button : bool
    question_number : Optional[Union[str, int, None]] = None
    error_occured: bool
    verified_button: bool 

    
class ChatItem(BaseModel):
    _id: str
    user_id: str
    credit_service_user_id: Optional[str] = None
    times: str
    date: str
    text: str
    sent_by_user: bool
    timestamp: Optional[datetime] = None


class ChatHistoryResponseSchema(BaseModel):
    chat_history : Optional[List[ChatItem]] = []
    message : str
