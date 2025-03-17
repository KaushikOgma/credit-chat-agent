import io
import os
import tempfile
from app.services.data_ingestion import process_folder
from app.schemas.module_schema import FolderPathRequest


async def extract_data(req_data: FolderPathRequest):
    try:
        data = req_data.input_data
        extractor = process_folder()
        extracted_info = await extractor.extract_information(data)
        return extracted_info
    except Exception as e:
        return {"error": str(e)}