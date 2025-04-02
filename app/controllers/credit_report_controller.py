import os
import sys
sys.path.append(os.getcwd())
import traceback
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
        print("User_id........", user_id)
        print("User_query........", user_query)
        try:
            if not self.vectorizer.vectordb:
                # Load the vector store
                self.vectorizer.load_vectorstore()
            # step-1: first need to check if there is data in mongo db for the user
            report = await self.credit_report_repo.get_todays_reoprt(db, user_id)
            # print(report)
            # print("get_credit_report_context: report:: ",len(report))
            if report:
                # There are report in the mongo that means we have the latest data
                if not report["isVectorized"]:
                    mongo_data, vector_data = await self.credit_report_processor_service.process_report(credit_report_json=None, user_id=user_id, categorized_resp=report["report"])   
                    print("credit report processed")
                    # Sync the vector DB with the latest QA pairs
                    await self.vectorizer.create_vectorstore(vector_data, "report_data_id", "topics")
                    print("credit report added to vector db")
                    await self.credit_report_repo.update_report(db, report["_id"], {"isVectorized": True})
                    print("Make credit report is vectorize true")
            else:
                # There are no report in the mongo db for today
                credit_report = await self.credit_report_extractor_service.get_credit_report(user_id)     
                if credit_report:
                    print("credit report found")
                    mongo_data, vector_data = await self.credit_report_processor_service.process_report(user_id=user_id, credit_report_json=credit_report, categorized_resp=None)   
                    if mongo_data is not None:
                        print("credit report processed")
                        mongo_data["isVectorized"] = False
                        inserted_id = await self.credit_report_repo.add_report(db, mongo_data)
                        print("credit report added:: ",inserted_id)
                        if not self.vectorizer.vectordb:
                            self.vectorizer.load_vectorstore()
                        await self.vectorizer.create_vectorstore(vector_data, "report_data_id", "topics")
                        print("credit report added to vector db")
                        await self.credit_report_repo.update_report(db, inserted_id, {"isVectorized": True})
                        print("Make credit report is vectorize true")
            # inserted_id = await self.credit_report_repo.add_report(db ,mongo_data) 
            context_list, score_list = await self.vectorizer.get_related_topics(user_id, user_query, top_context=3)
            print("get_credit_report_context:: context found - ",len(context_list))
            print("get_credit_report_context:: score_list found - ",score_list)
            if context_list is not None:
                combined_context = "; \n".join(elm for elm in context_list)
                if len(combined_context) > 5:
                    user_context = combined_context
            return user_context
        except Exception as error:
            print("get_credit_report_context:: error:: ",traceback.format_exc())
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return user_context
        

# import asyncio
# from app.repositories.credit_report_repositories import CreditReportRepository
# from app.services.credit_report_extractor import CreditReportExtractor
# from app.services.credit_report_processor import CreditReportProcessor
# from app.db import get_db_instance


# async def test_report():
#     try:
#         db = get_db_instance()

#         # Mock services
#         credit_report_extractor_service = CreditReportExtractor()
#         credit_report_processor_service = CreditReportProcessor()
#         credit_report_repo = CreditReportRepository()

#     # Initialize the controller
#         controller = CreditReportController(
#             credit_report_extractor_service=credit_report_extractor_service,
#             credit_report_processor_service=credit_report_processor_service,
#             credit_report_repo=credit_report_repo,
#         )

#         # Test data
#         user_id = "32b397c1-d160-44bc-9940-3d16542d8718"
#         user_query = "What is my credit score?"

#         # Call the method and print the result
#         try:
#             result = await controller.get_credit_report_context(db, user_id, user_query)
#             print("Result:", result)
#         except Exception as e:
#             print(f"[ERROR] Failed to fetch credit report context: {e}")
            
#     except Exception as e:
#         print(f"[ERROR] Main function failed: {e}")
#         traceback.print_exc()
#     finally:
#         # Ensure the database connection is closed
#         if db is not None:
#             db.client.close()

# if __name__ == "__main__":
#     asyncio.run(test_report())