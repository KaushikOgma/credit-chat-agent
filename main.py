import uvicorn
import socketio
import logging
# Import Libraries
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import (
    RequestValidationError,
    HTTPException,
    ResponseValidationError,
)
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from setuptools._distutils.util import strtobool
from app.utils.exceptions import ErrorHandler
from app.routes import router
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """
    Defines the lifespan of the FastAPI application.

    This function is used to perform startup and shutdown logic for the application.
    It yields to allow the application to run and then performs any necessary cleanup.

    Args:
        fastapi_app (FastAPI): The FastAPI application instance.
    """
    # Startup logic here
    setup_security_scheme(fastapi_app)
    yield  # This will allow the fastapi_app to run

    # Shutdown logic (if needed)
    # Any cleanup or shutdown logic can go here


# Initiate the FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    root_path=settings.PROJECT_ROOT_PATH,
    swagger_ui_parameters={"syntaxHighlight.theme": "arta-dark"},
    lifespan=lifespan,
)

# CORS Middleware Defined
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

# Define lifespan to replace @on_event de
def setup_security_scheme(fastapi_app: FastAPI):
    """
    Configures the security scheme for the FastAPI application.

    This function sets up the OpenAPI schema for the application, defining a BearerAuth security scheme.
    It then applies this security scheme globally to all routes in the application.

    Args:
        app (FastAPI): The FastAPI application instance.

    Returns:
        dict: The updated OpenAPI schema for the application.
    """
    try:
        fastapi_app.openapi_schema = fastapi_app.openapi()
        fastapi_app.openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",  # Use "http" as a string, not an enum
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT Bearer token authorization",
            }
        }

        # Apply the security scheme globally to all routes
        for path in fastapi_app.openapi_schema["paths"].values():
            for method in path.values():
                method["security"] = [{"BearerAuth": []}]
        return fastapi_app.openapi_schema
    except Exception as e:
        logger.exception(f"setup_security_scheme:: error - {str(e)}")


# Custom error handlers
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """
    Handles RequestValidationError exceptions by delegating to the ErrorHandler.

    Args:
        request (Request): The incoming request object.
        exc (RequestValidationError): The RequestValidationError exception.

    Returns:
        JSONResponse: The JSON response containing the status code and the error message.
    """
    return await ErrorHandler.request_validation_exception_handler(request, exc)


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(
    request: Request, exc: ResponseValidationError
):
    """
    Handles Response Validation exceptions by delegating to the ErrorHandler.

    Args:
        request (Request): The incoming request object.
        exc (ResponseValidationError): The Response Validation exception.

    Returns:
        JSONResponse: The JSON response containing the error message.
    """
    return await ErrorHandler.response_validation_exception_handler(request, exc)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handles HTTP exceptions by delegating to the ErrorHandler.

    Args:
        request (Request): The incoming request object.
        exc (HTTPException): The HTTP exception that occurred.

    Returns:
        JSONResponse: The JSON response containing the error message.
    """
    return await ErrorHandler.http_exception_handler(request, exc)


@app.get("/")
async def root():
    """
    Returns the root page of the application.

    This function handles the GET request to the root URL ("/") and returns an HTML response
    with the project name.

    Returns:
        HTMLResponse: The HTML response containing the project name.
    """
    return HTMLResponse(content=f"<center><h1>{settings.PROJECT_NAME}</h1></center>")

# Initialize Socket.IO server
sio = socketio.AsyncServer(
    async_mode="asgi",     # Use ASGI mode for FastAPI compatibility
    cors_allowed_origins="*",  # Allow all origins
    logger=True,
    engineio_logger=True,
    eio=3
)

# Wrap the FastAPI app with Socket.IO
sio_app = socketio.ASGIApp(sio, app)


# Initialize and include all routes
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:sio_app", host="0.0.0.0", port=int(settings.APP_PORT), reload=True)
