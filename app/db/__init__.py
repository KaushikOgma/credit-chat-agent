import os
from pymongo import MongoClient
from fastapi.exceptions import HTTPException
from pydantic import ValidationError
from fastapi.responses import JSONResponse
from pymongo.errors import PyMongoError
import pymongo
from datetime import datetime
from contextlib import asynccontextmanager

import urllib
from app.utils.config import settings
import time

# MONGO_URI = f"{settings.DB_PROTOCOL}://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}/?retryWrites=true&w=majority"

MONGODB_PORT = ":"+settings.DB_PORT if "mongodb+srv" not in settings.DB_PROTOCOL else ""
MONGO_URI = f"{settings.DB_PROTOCOL}://{urllib.parse.quote_plus(settings.DB_USER)}:{urllib.parse.quote_plus(settings.DB_PASSWORD)}@{settings.DB_HOST}{MONGODB_PORT}"

# print("MONGO_URI:: ", MONGO_URI)
# print("settings.DB_NAME:: ", settings.DB_NAME)


@asynccontextmanager
async def get_db():
    # Attempt to establish a connection to MongoDB
    client = MongoClient(MONGO_URI)
    try:
        # Access your MongoDB database
        db = client[settings.DB_NAME]
        # print("MONGO_URI:: ",MONGO_URI)
        yield db
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        client.close()


def get_db_instance():
    """
    **summary**
    Returns a MongoDB database instance.
    Attempts to establish a connection to MongoDB using the provided MONGO_URI.
    Accesses the specified database using the DB_NAME.

    Returns:
        db (pymongo.database.Database): A MongoDB database instance.

    Raises:
        Exception: If there is an error establishing a connection to MongoDB.
    """
    try:
        # Attempt to establish a connection to MongoDB
        client = MongoClient(MONGO_URI)
        # Access your MongoDB database
        db = client[settings.DB_NAME]
        return db
    except Exception as error:
        # error_message = "Database connection error: " + str(error)
        # print("get_db_instance:: error - " + error_message)
        # return JSONResponse(content={"message": error_message}, status_code=500)
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(error)}")

