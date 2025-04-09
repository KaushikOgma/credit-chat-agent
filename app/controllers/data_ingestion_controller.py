from fastapi.responses import JSONResponse
from app.controllers.evaluation_controller import EvaluationController
from app.controllers.finetune_controller import FinetuneController
from app.controllers.metadata_controller import MetadataController
from app.repositories.evaluation_repositories import EvaluationRepository
from app.repositories.metadata_repositories import MetadataRepository
from app.repositories.finetune_repositories import FinetuneRepository
from app.services.qa_evaluator import QAEvaluator
from app.services.qa_generator import QAGenerator
from app.services.data_ingestor import DataIngestor
from app.schemas.data_ingestion_schema import FileUploadRequest
from app.schemas.metadata_schema import MetadataSchema
from app.schemas.finetune_schema import TrainQASchema
from app.schemas.evaluation_schema import EvalQASchema
from app.services.data_ingestor import DataIngestor
from tqdm import tqdm
from pymongo.database import Database
from app.utils.config import settings
from app.utils.logger import setup_logger

logger = setup_logger()

class DataIngestionController:
    
    def __init__(self, metadata_repo: MetadataRepository, finetune_repo: FinetuneRepository,eval_repo: EvaluationRepository, finetune_controller: FinetuneController, eval_controller: EvaluationController, matadata_controller: MetadataController, data_ingestor: DataIngestor, qa_generator: QAGenerator, qa_evaluator: QAEvaluator):
        self.metadata_repo = metadata_repo
        self.finetune_repo = finetune_repo
        self.eveval_repo = eval_repo
        self.data_ingestor = data_ingestor
        self.qa_generator = qa_generator
        self.qa_evaluator = qa_evaluator
        self.finetune_controller = finetune_controller
        self.eval_controller = eval_controller
        self.matadata_controller = matadata_controller
        self.service_name = "data_ingestor_service"

    async def import_training_data(self, db: Database, req_data: FileUploadRequest):
        try:
            processed_files = {}
            files = req_data.files  
            extracted_info = await self.data_ingestor.ingest_files(files)
            for file_name, file_details in tqdm(extracted_info.items(), desc="Processing Files"):
                metadata_entry =  MetadataSchema(
                    fileName = file_name,
                    fileType = file_details["content_type"],
                    content = file_details["content"],
                    isTrainData = True,
                    isProcessed = False,
                )
                inserted_id = await self.matadata_controller.add_metadata(
                    data = metadata_entry.model_dump(),
                    db = db
                )
                processed_files[file_name] = inserted_id
            return processed_files
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error


    async def import_evaluation_data(self, db: Database, req_data: FileUploadRequest):
        try:
            processed_files = {}
            files = req_data.files  
            extracted_info = await self.data_ingestor.ingest_files(files)
            for file_name, file_details in tqdm(extracted_info.items(), desc="Processing Files"):

                metadata_entry =  MetadataSchema(
                    fileName = file_name,
                    fileType = file_details["content_type"],
                    content = file_details["content"],
                    isTrainData = False,
                    isProcessed = False,
                )
                inserted_id = await self.matadata_controller.add_metadata(
                    data = metadata_entry.model_dump(),
                    db = db
                )
                processed_files[file_name] = inserted_id
            return processed_files
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error