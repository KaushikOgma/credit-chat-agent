"""
This module contains all error handlers

Usage:
    from app.utils.error_handler import ErrorHandler

    async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
        return await ErrorHandler.request_validation_exception_handler(request, exc)

    async def http_exception_handler(request: Request, exc: HTTPException):
        return await ErrorHandler.http_exception_handler(request, exc)

    async def response_validation_exception_handler(request: Request, exc: ResponseValidationError):
        return await ErrorHandler.response_validation_exception_handler(request, exc)

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
        return await ErrorHandler.request_validation_exception_handler(request, exc)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return await ErrorHandler.http_exception_handler(request, exc)

    @app.exception_handler(ResponseValidationError)
    async def response_validation_exception_handler(request: Request, exc: ResponseValidationError):
        return await ErrorHandler.response_validation_exception_handler(request, exc)
"""

from fastapi import Request, HTTPException, status
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from app.utils.logger import setup_logger
logger = setup_logger()


class ErrorHandler:
    """
    A class that handles errors and exceptions in the application.

    It provides methods to format error messages and handle different types of exceptions.
    """

    @staticmethod
    def format_error_message(exc):
        """
        Formats an error message from a given exception.

        Args:
            exc: The exception object containing error details.

        Returns:
            A formatted string containing error messages.
        """
        try:
            details = exc.errors()
            error_list = []
            for error in details:
                err_msg = error["msg"]
                if "loc" in error:
                    locs = "->".join(str(item) for item in list(error["loc"]))
                    err_msg += " at " + locs
                error_list.append(err_msg)
            return ",".join(sorted(set(error_list)))
        except Exception as e:
            logger.exception(f"format_error_message:: error - {str(e)}")

    @staticmethod
    async def request_validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """
        A static method that handles Request Validation exceptions.

        Args:
            request (Request): The request object.
            exc (RequestValidationError): The Request Validation exception.

        Returns:
            JSONResponse: The JSON response containing the status code and the error message.
        """
        error_message = ErrorHandler.format_error_message(exc)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"message": error_message}),
        )

    @staticmethod
    async def response_validation_exception_handler(
        request: Request, exc: ResponseValidationError
    ):
        """
        A static method that handles Response Validation exceptions.

        Args:
            request (Request): The request object.
            exc (ResponseValidationError): The Response Validation exception.

        Returns:
            JSONResponse: The JSON response containing the status code and the error message.
        """
        error_message = ErrorHandler.format_error_message(exc)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"message": error_message}),
        )

    @staticmethod
    async def http_exception_handler(request: Request, exc: HTTPException):
        """
        A static method that handles HTTP exceptions.

        Args:
            request (Request): The request object.
            exc (HTTPException): The HTTP exception.

        Returns:
            JSONResponse: The JSON response containing the status code and the error message.
        """
        return JSONResponse(
            status_code=exc.status_code, content={"message": exc.detail}
        )
