from typing import List
from pydantic import BaseModel
from fastapi import UploadFile


class ChatRequest(BaseModel):
    question: str

class ChatSchema(BaseModel):
    question: str
    answer: str

class ChatResponse(BaseModel):
    data: ChatSchema
    message: str