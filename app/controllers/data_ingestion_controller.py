from app.services.data_ingestor import process_folder
from app.schemas.module_schema import FolderPathRequest
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()


async def extract_data(req_data: FolderPathRequest):
    serviceName = "data_ingestor"
    try:
        data = req_data.input_data
        extractor = process_folder()
        extracted_info = await extractor.extract_information(data)
        return extracted_info
    except Exception as error:
        logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": serviceName})
        return {"error": str(error)}