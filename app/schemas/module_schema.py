from pydantic import BaseModel

class QAGenerateRequest(BaseModel):
    text: str
