from app.services.data_ingestor import DataIngestor
from app.schemas.module_schema import FileUploadRequest
from app.utils.config import settings
from app.utils.logger import setup_logger

logger = setup_logger()

async def import_data(req_data: FileUploadRequest):
    service_name = "data_ingestor"
    try:
        files = req_data.files  
        ingestor = DataIngestor()
        extracted_info = await ingestor.ingest_files(files)
        
        return extracted_info
    except Exception as error:
        logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": service_name})
        return {"error": str(error)}
