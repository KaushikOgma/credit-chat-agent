from pydantic import BaseModel

class QAGenerateRequest(BaseModel):
    text: str

class FolderPathRequest(BaseModel):
    folder_path: str