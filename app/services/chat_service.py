import os
import sys
sys.path.append(os.getcwd())
import asyncio
from typing import List, Union, Dict, Any
from app.utils.helpers.prompt_helper import chat_system_content_message
from app.utils.config import settings
import openai
import urllib.parse
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from langgraph.graph import StateGraph, START
from fastapi import HTTPException
from pymongo.errors import PyMongoError
from app.controllers.credit_report_controller import CreditReportController
from app.services.credit_report_extractor import CreditReportExtractor
from app.services import credit_report_extractor, credit_report_processor
from app.repositories.credit_report_repositories import CreditReportRepository
from app.utils.logger import setup_logger
from app.dependencies.chat_report_dependencies import get_credit_report_controller

logger = setup_logger()

# ✅ Correct MongoDB Initialization
mongo_client = AsyncIOMotorClient(
    f"{settings.DB_PROTOCOL}://{urllib.parse.quote_plus(settings.DB_USER)}:{urllib.parse.quote_plus(settings.DB_PASSWORD)}@{settings.DB_HOST}"
)
db = mongo_client[settings.DB_NAME]

class ToolFunction:
    def __init__(self):
        self.chat_limit = settings.CHAT_HISTORY_LIMIT
        self.mongo_uri = f"{settings.DB_PROTOCOL}://{urllib.parse.quote_plus(settings.DB_USER)}:{urllib.parse.quote_plus(settings.DB_PASSWORD)}@{settings.DB_HOST}"
        self.client = AsyncIOMotorClient(self.mongo_uri)
        self.db = self.client[settings.DB_NAME]
        self.chat_collection = self.db["chat_history"]

    async def fetch_previous_chat(self, state: Dict[str, Any]) -> list:
        user_id = state.get("user_id")  # Extract only user_id from the state
        print(f"[DEBUG] Fetching previous chat for user_id: {user_id}")

        if not user_id:
            print("[ERROR] Missing user_id in fetch_previous_chat input")
            return []
        
        chat_data = await self.chat_collection.find_one({"user_id": user_id})
        result = chat_data.get("chat_history", [])[-self.chat_limit:] if chat_data else []
        
        print(f"[DEBUG] Previous chat fetched: {result}")
        return result


    async def store_chat(self, input_data: Dict[str, Any]):
        user_query = input_data.get("user_query")
        bot_response = input_data.get("bot_response")
        print(f"[DEBUG] Storing chat: user_query='{user_query}', bot_response='{bot_response}'")
        if user_query is None or bot_response is None:
            raise ValueError("Missing user_query or bot_response in store_chat")
        await self.chat_collection.update_one(
            {"user_id": input_data["user_id"]},
            {"$push": {"chat_history": {"user_query": user_query, "bot_response": bot_response}}},
            upsert=True
        )
        print("[DEBUG] Chat stored successfully")

    async def get_user_query_and_id(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Returns both user_id and user_query."""
        print(f"[DEBUG] Extracting user query and ID: {input_data}")
        required_keys = ["user_id", "user_query"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Missing required key: {key} in input_data")
        response = {
            "user_id": input_data["user_id"],
            "user_query": input_data["user_query"]
        }
        print(f"[DEBUG] get_user_query_and_id response: {response}")
        return response

    async def get_user_query(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """Returns only the user query."""
        print(f"[DEBUG] Extracting user query: {input_data}")
        if "user_query" not in input_data:
            raise ValueError("Missing required key: user_query in input_data")
        response = {"user_query": input_data["user_query"]}
        print(f"[DEBUG] get_user_query response: {response}")
        return response

    async def get_user_id(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """Returns only the user ID."""
        print(f"[DEBUG] Extracting user ID: {input_data}")
        if "user_id" not in input_data:
            raise ValueError("Missing required key: user_id in input_data")
        response = {"user_id": input_data["user_id"]}
        print(f"[DEBUG] get_user_id response: {response}")
        return response


    def merge_context(self, data_list: List[Dict[str, Any]]):
        """Merge multiple dictionaries into a single context dictionary."""
        if not isinstance(data_list, list):
            print(f"[DEBUG] Unexpected data type in merge_context: {type(data_list)} -> {data_list}")
            return {}

        merged_data = {}
        for data in data_list:
            if isinstance(data, dict):
                merged_data.update(data)
            else:
                print(f"[DEBUG] Skipping unexpected data type in merge_context: {type(data)} -> {data}")
        
        print(f"[DEBUG] Merged Context: {merged_data}")  # Debugging log
        return merged_data

class ChatService:
    """
    A class to evaluate the quality of the AI-generated responses.
    """
    def __init__(self):
        # Initialize the Vectorizer Engine
        self.temperature = 0.7
        self.max_tokens = 500
        self.openai = openai
        self.openai.api_key = settings.OPENAI_API_KEY
        self.client = self.openai.Client()  # Create a client instance
        self.similarity_threshold = 0.8
        self.system_prompt = chat_system_content_message()
        self.service_name = "chat_service"
        self.model_id = "ft:gpt-4o-2024-08-06:the-great-american-credit-secret-llc::BABaftly"

    async def get_response(self, question: str) -> Union[str, None]:
        try:
            # system_propmt = "You are a Credit Genius Assistant."
            response = await asyncio.to_thread(
                self.client.chat.completions.create,  # Correct method for OpenAI >= 1.0
                model=self.model_id,
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
        
async def store_chat_async(input_data):
    print(f"[DEBUG] Async store chat called with: {input_data}")
    tool = ToolFunction()
    await tool.store_chat(input_data)

async def credit_report_context_wrapper(state):
    controller = get_credit_report_controller()
    return await controller.get_credit_report_context(db, state["user_id"], state["user_query"])

def setup_graph():
    print("[DEBUG] Setting up graph")  # Debug
    graph = StateGraph(Dict[str, Any])
    tool = ToolFunction()
    chat_service = ChatService()

    # Step 1: Extract user query and user ID separately
    graph.add_node("get_user_query_and_id", tool.get_user_query_and_id)
    graph.add_node("get_user_query", tool.get_user_query)
    graph.add_node("get_user_id", tool.get_user_id)

    # Step 2: Check for credit report if verified
    graph.add_node("check_recent_data", credit_report_context_wrapper)

    # Step 3: Fetch previous chat history
    graph.add_node("fetch_previous_chat", tool.fetch_previous_chat)

    # Step 4: Merge the fetched context
    graph.add_node("merge_context", tool.merge_context)

    # Step 5: Process query with AI agent
    graph.add_node("agent1_process", chat_service.get_response)

    # Step 6: Store chat history
    graph.add_node("store_chat", store_chat_async)

    # Conditional check for verification
    def check_verification(state):
        print("[DEBUG] Checking verification for state:", state)  # Debug
        if state.get("is_verified", False):
            return "check_recent_data"  # Fetch credit report only if verified
        return "fetch_previous_chat"  # Otherwise, fetch previous chat


    # Graph edges
    graph.add_edge(START, "get_user_query_and_id")

    graph.add_edge("get_user_query_and_id", "get_user_query")  # Extract query
    graph.add_edge("get_user_query_and_id", "get_user_id")  # Extract user ID

    graph.add_edge("get_user_id", "fetch_previous_chat")  # Fetch previous chat using user_id
    graph.add_edge("get_user_query_and_id", "check_recent_data")  # Pass both user_id & user_query for credit check
    graph.add_edge("get_user_query", "merge_context")  # Pass user_query directly to merge

    graph.add_conditional_edges("get_user_query_and_id", check_verification)

    graph.add_edge("check_recent_data", "merge_context")  # Ensure credit report data is passed
    graph.add_edge("fetch_previous_chat", "merge_context")  # Ensure chat history is passed

    graph.add_edge("merge_context", "agent1_process")
    graph.add_edge("agent1_process", "store_chat")

    print("[DEBUG] Graph setup completed")  # Debug
    return graph


async def main():
    graph = setup_graph()
    runnable_graph = graph.compile()
    test_input = {
        "user_id": "32b397c1-d160-44bc-9940-3d16542d8718",
        "user_query": "What is my credit score?",
        "is_verified": False
    }
    print(f"[DEBUG] Running graph with input: {test_input}")
    result = await runnable_graph.ainvoke(test_input)
    print("[DEBUG] Final Output:", result)


if __name__ == "__main__":
    asyncio.run(main())




# async def get_chat_response():
#     # Initialize the fine-tuner
#     chat_service = ChatService()
#     model_id = "ft:gpt-4o-2024-08-06:the-great-american-credit-secret-llc::BABaftly"
#     # Sample Q&A data for testing
#     qa_pairs = [
#         {
#             "question_id": "f3b82a10-26d2-4c8f-b4b8-b71f3d3a80a4",
#             "question": "What exactly is credit?",
#             "answer": "It's the system of borrowing money with the agreement to pay it back later, often with interest."
#         },
#         {
#             "question_id": "7e1e5a5a-6e6d-4a67-9d38-2d5f10b3a1b5",
#             "question": "Why should I care about credit?",
#             "answer": "Good credit can help you get loans with better interest rates and is essential for big purchases like a car or home."
#         },
#         {
#             "question_id": "c8b3b0cb-d28b-49a9-95d5-bd6a7c0e4b2d",
#             "question": "What’s a credit score?",
#             "answer": "A number that lenders use to determine how risky it is to lend you money, based on your credit history."
#         },
#         {
#             "question_id": "9d7a5cb5-d6b2-4a1c-8c58-f845db22cf84",
#             "question": "How can I improve my credit score?",
#             "answer": "Make payments on time, keep your credit card balances low, and manage your debts wisely."
#         },
#         {
#             "question_id": "0f4322b9-b7f2-4b1b-bbd9-92a5b5d1b7d5",
#             "question": "Can someone my age have a credit score?",
#             "answer": "Yes, once you turn 18 and start using credit products, like a credit card or loan, you begin to build a credit history."
#         },
#         {
#             "question_id": "b18a9e83-43bd-4b78-8f68-1e6c3c5d9b22",
#             "question": "What’s on a credit report?",
#             "answer": "Details of your credit accounts, payment history, debts, and sometimes employment history, used to calculate your score."
#         }
#     ]
#     response = await chat_service.get_response(qa_pairs[0]["question"], model_id)
#     print("response:: ",response)

# if __name__ == "__main__":
#     asyncio.run(get_chat_response())
