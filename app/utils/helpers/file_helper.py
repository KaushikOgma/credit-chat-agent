"""
This Module is responsible for Managing
all of the file actions for the application

Usage:
    from app.utils.helpers.file_helper import save_file

    file_path = save_file(file)
"""

import os
import shutil
from fastapi import UploadFile
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()


def save_file(
    upload_file: UploadFile, upload_dir: str = settings.LOCAL_UPLOAD_LOCATION
) -> str:
    """
    Save an uploaded file to the specified directory.

    Args:
        upload_file (UploadFile): The file to be uploaded.
        upload_dir (str): The directory to save the uploaded file.

    Returns:
        str: The file path where the file is saved.
    """
    try:
        # Ensure the upload directory exists
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        # Construct the file path
        file_location = os.path.join(".", upload_dir, upload_file.filename)
        # Save the file
        with open(file_location, "wb") as file_object:
            shutil.copyfileobj(upload_file.file, file_object)

        return file_location
    except Exception as e:
        logger.exception(f"Failed to save file: {str(e)}")
        raise e


def delete_file(file_path: str):
    """
    Delete a file from the file system.

    Args:
        file_path (str): The path of the file to delete.
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.exception(f"Failed to delete file: {str(e)}")
        return None


# Export the required function
__all__ = ["save_file", "delete_file"]
