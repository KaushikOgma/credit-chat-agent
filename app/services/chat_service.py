import os
import sys
sys.path.append(os.getcwd())
import asyncio
from typing import List, Union, Dict, Any, Annotated, TypedDict
from app.utils.helpers.prompt_helper import chat_system_content_message
from app.utils.config import settings
import openai
import urllib.parse
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from langgraph.graph import StateGraph, START
from app.dependencies.chat_report_dependencies import get_credit_report_controller
from app.utils.logger import setup_logger
import operator

logger = setup_logger()

class ChatService:
    def __init__(self):
        self.temperature = 0.7
        self.max_tokens = 500
        self.openai = openai
        self.openai.api_key = settings.OPENAI_API_KEY
        self.client = self.openai.Client()
        self.similarity_threshold = 0.8
        self.system_prompt = chat_system_content_message()
        self.model_id = 'ft:gpt-4o-2024-08-06:the-great-american-credit-secret-llc::BABaftly'
        self.service_name = "chat_service"

    async def get_response(self, question: str, model_id: str) -> Union[str, None]:
        try:
            # system_propmt = "You are a Credit Genius Assistant."
            response = await asyncio.to_thread(
                self.client.chat.completions.create,  # Correct method for OpenAI >= 1.0
                model=model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return None
    


async def get_chat_response():
    # Initialize the fine-tuner
    chat_service = ChatService()
    model_id = "ft:gpt-4o-2024-08-06:the-great-american-credit-secret-llc::BABaftly"
    # Sample Q&A data for testing
    qa_pairs = [
        {
            "question_id": "f3b82a10-26d2-4c8f-b4b8-b71f3d3a80a4",
            "question": "What exactly is credit?",
            "answer": "It's the system of borrowing money with the agreement to pay it back later, often with interest."
        },
        {
            "question_id": "7e1e5a5a-6e6d-4a67-9d38-2d5f10b3a1b5",
            "question": "Why should I care about credit?",
            "answer": "Good credit can help you get loans with better interest rates and is essential for big purchases like a car or home."
        },
        {
            "question_id": "c8b3b0cb-d28b-49a9-95d5-bd6a7c0e4b2d",
            "question": "What’s a credit score?",
            "answer": "A number that lenders use to determine how risky it is to lend you money, based on your credit history."
        },
        {
            "question_id": "9d7a5cb5-d6b2-4a1c-8c58-f845db22cf84",
            "question": "How can I improve my credit score?",
            "answer": "Make payments on time, keep your credit card balances low, and manage your debts wisely."
        },
        {
            "question_id": "0f4322b9-b7f2-4b1b-bbd9-92a5b5d1b7d5",
            "question": "Can someone my age have a credit score?",
            "answer": "Yes, once you turn 18 and start using credit products, like a credit card or loan, you begin to build a credit history."
        },
        {
            "question_id": "b18a9e83-43bd-4b78-8f68-1e6c3c5d9b22",
            "question": "What’s on a credit report?",
            "answer": "Details of your credit accounts, payment history, debts, and sometimes employment history, used to calculate your score."
        }
    ]
    response = await chat_service.get_response(qa_pairs[0]["question"], model_id)
    print("response:: ",response)

if __name__ == "__main__":
    asyncio.run(get_chat_response())
