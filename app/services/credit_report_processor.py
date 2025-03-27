import asyncio
import json
import re
import traceback
from typing import List, Union
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from app.services.pinecone_vectorizer import OpenAIEmbedding, VectorizerEngine
from app.utils.helpers.prompt_helper import credit_report_process_conversation_messages
from langchain_openai import ChatOpenAI
from app.utils.config import settings
from app.services.chat_service import ChatService
import openai
from tenacity import (
    retry,
    wait_random_exponential,
)  
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from app.utils.logger import setup_logger
logger = setup_logger()

class ArrayReportProcessor:
    def __init__(self):
        self.service_name = "credit_report_processor"
        self.credit_liability_filter = "CREDIT_LIABILITY"
        self.credit_liability_categorizer = "@_AccountStatusType"
        self.credit_inquiry_filter = "CREDIT_INQUIRY"
        self.credit_inquiry_categorizer = "@_PurposeType"
        self.credit_score_filter = "CREDIT_SCORE"
        self.credit_score_factor = "_FACTOR"
        self.credit_summary_filter = "CREDIT_SUMMARY"
        self.credit_summary_dataset = "_DATA_SET"
        self.service_name = "credit_report_processor"

    async def process_credit_liabilities(self, credit_liabilities: list[dict]):
        liabilities = {}
        try:
            for elm in credit_liabilities:
                sub_cat = f"{elm[self.credit_liability_categorizer].upper()}_{self.credit_liability_filter}"
                if sub_cat in liabilities:
                    liabilities[sub_cat].append(elm)
                else:
                    liabilities[sub_cat] = [elm]
            return liabilities
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            liabilities[self.credit_liability_filter] = None
            return liabilities


    async def process_credit_inquiries(self, credit_inquiries: list[dict]):
        inquiries = {}
        try:
            for elm in credit_inquiries:
                sub_cat = f"{elm[self.credit_inquiry_categorizer].upper()}_{self.credit_inquiry_filter}"
                if sub_cat in inquiries:
                    inquiries[sub_cat].append(elm)
                else:
                    inquiries[sub_cat] = [elm]
            return inquiries
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            inquiries[self.credit_inquiry_filter] = None
            return inquiries


    async def process_credit_score(self, credit_score: list[dict]):
        processed_credit_score = []
        try:
            for elm in credit_score:
                try:
                    processed_factors = []
                    factors = elm[self.credit_score_factor]
                    for fact in factors:
                        processed_factors.append(fact["@_Text"])
                    elm[self.credit_score_factor] = processed_factors
                    processed_credit_score.append(elm)
                except Exception as error:
                    processed_credit_score.append(elm)
            return processed_credit_score
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return credit_score


    async def process_credit_summary(self, credit_summary: dict):
        processed_credit_summary = credit_summary.copy()
        try:
            data_sets = credit_summary[self.credit_summary_dataset]
            dataset_dict = {}
            for elm in data_sets:
                try:
                    dataset_dict[elm["@_Name"].split(":")[-1]] = elm["@_Value"]
                except Exception as error:
                    continue
            processed_credit_summary[self.credit_summary_dataset] = dataset_dict
            return processed_credit_summary
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return processed_credit_summary



    async def collect_categories(self, credit_report_json: dict):
        try:
            categories = {"METADATA": {}}
            if "CREDIT_RESPONSE" in credit_report_json:
                for key, value in credit_report_json["CREDIT_RESPONSE"].items():
                    if "@" in key:
                        categories["METADATA"][key] = value
                    else:
                        if key in [self.credit_liability_filter]:
                            credit_liabilities = await self.process_credit_liabilities(value)
                            for category, liability in credit_liabilities.items():
                                categories[category] = liability
                        elif key in [self.credit_inquiry_filter]:
                            credit_inquiries = await self.process_credit_inquiries(value)
                            for category, inquiry in credit_inquiries.items():
                                categories[category] = inquiry
                        elif key in [self.credit_score_filter]:
                            credit_score = await self.process_credit_score(value)
                            categories[key] = credit_score
                        elif self.credit_summary_filter in key:
                            credit_summary = await self.process_credit_summary(value)
                            categories[key] = credit_summary
                        else:
                            categories[key.strip("_")] = value
            return categories
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return None
        


class CreditReportProcessor:
    """
    A class to evaluate the quality of the AI-generated responses.
    """
    def __init__(self):
        # Initialize the OpenAI Embedding
        self.encoder = OpenAIEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)
        # Initialize the Vectorizer Engine
        self.temperature = 0
        self.max_tokens = 500
        self.model_rate_limits = 2000
        self.max_concurent_request = int(self.model_rate_limits * 0.75)
        self.throttler = asyncio.Semaphore(self.max_concurent_request)
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
        self.similarity_threshold = 0.8
        self.service_name = "credit_report_processor"

    # Run Model Function
    @retry(wait=wait_random_exponential(min=15, max=40))
    async def run_model(self, messages):
        """
        Asynchronously runs the chat model while respecting token limits.

        Args:
            messages (list): List of input messages for the model.

        Returns:
            str: Model-generated output text.
        """
        try:
            model = ChatOpenAI(temperature=self.temperature, model=self.model_id)

            try:
                async with self.throttler:
                    output = await model._agenerate(messages, response_format={"type": "json_object"})
            except (openai.RateLimitError, openai.APIConnectionError) as error1:
                logger.exception(error1, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
                raise error1
            except Exception as error2:
                logger.exception(error2, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
                return 'ERROR'

            return output.generations[0].text.strip()
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return 'ERROR'
    

    async def parse_json_from_text(self, text):
        try:
            # Extract JSON using regex (handles Markdown formatting)
            json_match = re.search(r"\{.*\}", text, re.DOTALL)  # Match the first JSON block
            if json_match:
                json_data = json_match.group(0)  # Extract JSON part
                try:
                    return json.loads(json_data)  # Convert to Python dictionary
                except json.JSONDecodeError:
                    print("Error: Extracted JSON is invalid")
                    return None
            else:
                print("Error: No JSON found in response")
                return None
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return None
        

    async def get_response(self, credit_report_json: dict, category: str) -> Union[str, None]:
        try:
            messages = credit_report_process_conversation_messages(credit_report_json, category)
            response_content = await self.run_model(messages)
            response_content = await self.parse_json_from_text(response_content)
            return response_content
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return None
        

    async def get_processed_report(self, credit_report_json: dict):
        try:
            enhanced_report = {}

            async def process_category(category, category_data):
                """Helper function to process a single category."""
                processed_chunks = await self.get_response(category_data, category)
                return category, processed_chunks

            # Run multiple get_response calls concurrently using asyncio.create_task
            tasks = [asyncio.create_task(process_category(category, data)) for category, data in credit_report_json.items()]
            results = await asyncio.gather(*tasks)

            # Collect results
            for category, processed_chunks in results:
                enhanced_report[category] = processed_chunks

            return enhanced_report

        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return None


async def start_processing():
    # Initialize the fine-tuner
    credit_report_processor = CreditReportProcessor()
    report_processor = ArrayReportProcessor()
    credit_report_json_path = os.path.join(".", settings.LOCAL_UPLOAD_LOCATION ,"array_data.json")
    processed_credit_report_json_path = os.path.join(".", settings.LOCAL_UPLOAD_LOCATION ,"processed_array_data.json")
    credit_report_json = {}
    with open(credit_report_json_path, "r") as f:
        credit_report_json = json.load(f)
    # resp = await credit_report_processor.get_processed_report(credit_report_json, user_id)
    categorize_resp = await report_processor.collect_categories(credit_report_json)
    categorize_resp_org = {**categorize_resp}
    # print("categorize_resp:: \n",json.dumps(categorize_resp, indent=3))
    enhanced_resp = await credit_report_processor.get_processed_report(categorize_resp)
    for cat, data in enhanced_resp.items():
        enhanced_resp[cat] = {**enhanced_resp[cat], "raw_data": categorize_resp_org[cat]}
    print("enhanced_resp:: \n",json.dumps(enhanced_resp, indent=3))
    with open(processed_credit_report_json_path, "w") as f:
        json.dump(enhanced_resp, f, indent=3)


if __name__ == "__main__":
    asyncio.run(start_processing())
