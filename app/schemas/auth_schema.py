"""
This module containes all schemas those are related to auth add/update/read/delete requests.
"""
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from fastapi import status
from fastapi.responses import JSONResponse
from fastapi import Form
from fastapi import HTTPException
from fastapi.security import OAuth2PasswordRequestForm

# Credential exceptions error
invalid_credential_resp = JSONResponse(
    status_code=status.HTTP_401_UNAUTHORIZED,
    content={"message": "Invalid credentials"},
    headers={"WWW-Authenticate": "Bearer"},
)

credential_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Unauthenticated"
)

# Customized OAuth2 form model
class OAuth2PasswordRequestFormExtended(OAuth2PasswordRequestForm):
    """**Summary:**
    This class contains the schema of each login request form
    """
    def __init__(
        self,
        username: str = Form(),
        password: str = Form(default=None),
    ):
        self.username = username
        self.password = password


class LoginRequest(BaseModel):
    """**Summary:**
    This class contains the schema of each login request
    """
    password: str = Field(description="login password")
    email: str = Field(description="login email")


class AccessTokenResponse(BaseModel):
    """**Summary:**
    This class contains the schema of each login response
    """
    access_token: str = Field(description="jwt access token")
    status: str

class ChangePasswordRequest(BaseModel):
    """**Summary:**
    This class contains the schema for change password request
    """
    old_password: str = Field(description="Current password")
    new_password: str = Field(description="New password")




