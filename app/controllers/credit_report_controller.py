from typing import Union
from fastapi.responses import JSONResponse
import openai
from app.repositories.credit_report_repositories import CreditReportRepository
from app.services.credit_report_extractor import CreditReportExtractor
from app.services.credit_report_processor import CreditReportProcessor
from app.services.pinecone_vectorizer import VectorizerEngine, OpenAIEmbedding
from tqdm import tqdm
from pymongo.database import Database
from app.utils.config import settings
from app.utils.logger import setup_logger

logger = setup_logger()

class CreditReportController:
    
    def __init__(
            self, 
            credit_report_extractor_service: CreditReportExtractor, 
            credit_report_processor_service: CreditReportProcessor, 
            credit_report_repo: CreditReportRepository,

        ):
        self.credit_report_extractor_service = credit_report_extractor_service
        self.credit_report_processor_service = credit_report_processor_service
        self.credit_report_repo = credit_report_repo
        self.encoder = OpenAIEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)
        self.temperature = 0
        self.max_tokens = 500
        self.openai = openai
        self.model_id = settings.BASE_MODEL
        self.openai.api_key = settings.OPENAI_API_KEY
        self.client = self.openai.Client()  # Create a client instance
        self.vectorizer = VectorizerEngine(
            encoder=self.encoder,
            vector_db_name=settings.VECTOR_DB_NAME,
            batch_size=10,
            dimension=settings.VECTOR_DIMENSION,
            namespace="credit_reports"
        )
        self.service_name = "credit_report_service"

    async def get_credit_report_context(self, db: Database, user_id: str, user_query: str) -> Union[None, str]:
        user_context = None
        try:
            # step-1: first need to check if there is data in mongo db for the user
            report = await self.credit_report_repo.get_todays_reoprt(user_id)
            if report:
                # There are report in the mongo that means we have the latest data
                pass
            else:
                # There are no report in the mongo db for today
                credit_report = await self.credit_report_extractor_service.get_credit_report(user_id)     
                if credit_report:
                    mongo_data, vector_data = await self.credit_report_processor_service.process_report(credit_report, user_id)   
                    inserted_id = self.credit_report_repo.add_report(db, mongo_data)
                    if not self.vectorizer.vectordb:
                        self.vectorizer.load_vectorstore()
                    self.vectorizer.create_vectorstore(vector_data, "report_data_id", "topics")
            # inserted_id = await self.credit_report_repo.add_report(db ,mongo_data) 
            context_list, score_list = self.vectorizer.get_related_topics(user_id, user_query, top_context=3)
            if context_list is not None:
                combined_context = "; \n".join(elm for elm in context_list)
                if len(combined_context) > 5:
                    user_context = combined_context
            return user_context
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return user_context

