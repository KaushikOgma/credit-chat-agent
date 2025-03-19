import os
from pymongo import MongoClient
from fastapi.exceptions import HTTPException
from pydantic import ValidationError
from fastapi.responses import JSONResponse
from app.utils.config import settings
import time

# MONGO_URI = f"{settings.DB_PROTOCOL}://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}/?retryWrites=true&w=majority"
MONGO_URI = f"{settings.DB_PROTOCOL}://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}"

# print("MONGO_URI:: ", MONGO_URI)
# print("settings.DB_NAME:: ", settings.DB_NAME)


def get_db():
    """
    **summary**
    Returns a MongoDB database instance.
    This function establishes a connection to a MongoDB server using the provided `MONGO_URI` and retrieves the specified database. It uses the `pymongo` library to create a `MongoClient` object and connects to the MongoDB server. The `MongoClient` object is used to access the specified database by indexing into the `client` object with the `DB_NAME` value.
    The function uses a generator to yield the database instance. This allows the caller to use the database instance within a context manager, ensuring that the connection is properly closed after use.
    If an exception occurs during the connection or retrieval of the database, the exception is raised.

    Returns:
        db (pymongo.database.Database): A MongoDB database instance.

    Raises:
        Exception: If there is an error establishing a connection to MongoDB or retrieving the database.
    """
    start_time = time.time()  # Capture start time

    # Attempt to establish a connection to MongoDB
    client = MongoClient(MONGO_URI)
    # Access your MongoDB database
    db = client[settings.DB_NAME]

    end_time = time.time()  # Capture end time

    total_time = end_time - start_time

    # Get minutes and remaining seconds
    minutes = int(total_time // 60)  # Floor division for whole minutes
    seconds = total_time % 60  # Remainder represents seconds


    # print(f"Total time taken: {minutes} minutes and {seconds:.2f} seconds")
    try:
        # print("MONGO_URI:: ",MONGO_URI)
        yield db
    except Exception as e:
        raise e
    finally:
        db.client.close()


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
        error_message = "Database connection error: " + str(error)
        # print("get_db_instance:: error - " + error_message)
        return JSONResponse(content={"message": error_message}, status_code=500)
