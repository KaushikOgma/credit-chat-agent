from fastapi.responses import JSONResponse
from app.repositories.metadata_repositories import MetadataRepository
from app.services.data_ingestor import DataIngestor
from app.schemas.data_ingestion_schema import FileUploadRequest
from app.schemas.metadata_schema import MetadataSchema
from app.services.data_ingestor import DataIngestor
from pymongo.database import Database
from app.utils.config import settings
from app.utils.logger import setup_logger

logger = setup_logger()

class DataIngestionController:
    
    def __init__(self, metadata_repo: MetadataRepository, data_ingestor: DataIngestor):
        self.metadata_repo = metadata_repo
        self.data_ingestor = data_ingestor
        self.service_name = "data_ingestor"

    async def import_training_data(self, db: Database, req_data: FileUploadRequest):
        try:
            processed_files = {}
            files = req_data.files  
            extracted_info = await self.data_ingestor.ingest_files(files)
            for file_name, file_details in extracted_info.items():
                meatdata_entry = MetadataSchema(
                    fileName = file_name,
                    fileType = file_details["content_type"],
                    content = file_details["content"],
                    isTrainData = True,
                    isProcessed = False
                )
                inserted_id = await self.metadata_repo.add_metadata(db, meatdata_entry.model_dump())
                processed_files[file_name] = inserted_id
            return JSONResponse(
                        status_code=200, content={"data": processed_files, "message": "Data Inserted successfully"}
                    )
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error


    async def import_evaluation_data(self, db: Database, req_data: FileUploadRequest):
        try:
            processed_files = {}
            files = req_data.files  
            extracted_info = await self.data_ingestor.ingest_files(files)
            for file_name, file_details in extracted_info.items():
                meatdata_entry = MetadataSchema(
                    fileName = file_name,
                    fileType = file_details["content_type"],
                    content = file_details["content"],
                    isTrainData = False,
                    isProcessed = False
                )
                inserted_id = await self.metadata_repo.add_metadata(db, meatdata_entry.model_dump())
                processed_files[file_name] = inserted_id
            return JSONResponse(
                        status_code=200, content={"data": processed_files, "message": "Data Inserted successfully"}
                    )
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error