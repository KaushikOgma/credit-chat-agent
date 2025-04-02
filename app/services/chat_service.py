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
from langgraph.graph import StateGraph, START
from app.dependencies.chat_report_dependencies import get_credit_report_controller
from app.utils.logger import setup_logger
import operator

logger = setup_logger()

# MongoDB Initialization
mongo_client = AsyncIOMotorClient(
    f"{settings.DB_PROTOCOL}://{urllib.parse.quote_plus(settings.DB_USER)}:{urllib.parse.quote_plus(settings.DB_PASSWORD)}@{settings.DB_HOST}"
)
db = mongo_client[settings.DB_NAME]

class ToolFunction:
    def __init__(self):
        self.chat_limit = settings.CHAT_HISTORY_LIMIT
        self.chat_collection = db["chat_history"]

    async def fetch_previous_chat(self, state: Dict[str, Any]) -> list:
        user_id = state.get("user_id")
        if not user_id:
            raise ValueError("Missing user_id in fetch_previous_chat")
        chat_data = await self.chat_collection.find_one({"user_id": user_id})
        return chat_data.get("chat_history", [])[-self.chat_limit:] if chat_data else []

    async def store_chat(self, user_id: str, user_query: str, bot_response: str) -> Dict[str, Any]:
        """
        Stores the chat in the database and ensures the bot_response is passed through.
        """
        print(f"[DEBUG] Storing chat: user_query={user_query}, bot_response={bot_response}")  # Debugging

        # Ensure bot_response is always returned
        if not user_query or bot_response is None:
            print(f"[DEBUG] Skipping chat storage due to missing data: user_query={user_query}, bot_response={bot_response}")
            return {"bot_response": bot_response}  # Return the bot_response directly

        try:
            await self.chat_collection.update_one(
                {"user_id": user_id},
                {"$push": {"chat_history": {"user_query": user_query, "bot_response": bot_response}}},
                upsert=True
            )
        except Exception as e:
            print(f"[ERROR] Failed to store chat in database: {e}")
            logger.error(f"Failed to store chat in database: {e}")

        # Always return the bot_response
        return {"bot_response": bot_response}

    async def get_user_query_and_id(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[DEBUG] Input to get_user_query_and_id: {input_data}")  # Debug
        required_keys = ["user_id", "user_query"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Missing required key: {key} in input_data")
        return {"user_id": input_data["user_id"], "user_query": input_data["user_query"]}

    async def get_user_query(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        if "user_query" not in input_data:
            raise ValueError("Missing required key: user_query in input_data")
        return {"user_query": input_data["user_query"]}

    async def get_user_id(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        if "user_id" not in input_data:
            raise ValueError("Missing required key: user_id in input_data")
        return {"user_id": input_data["user_id"]}

    def merge_context(self, data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple dictionaries into a single context dictionary."""
        print(f"[DEBUG] Input to merge_context: {data}")
        if isinstance(data, dict):
            # If a single dictionary is passed, wrap it in a list
            data = [data]
        if not isinstance(data, list):
            raise ValueError("merge_context expects a list of dictionaries")
        merged_data = {}
        for item in data:
            if isinstance(item, dict):
                merged_data.update(item)
        print(f"[DEBUG] Output from merge_context: {merged_data}")
        return merged_data

class ChatService:
    def __init__(self):
        self.temperature = 0.7
        self.max_tokens = 500
        self.openai = openai
        self.openai.api_key = settings.OPENAI_API_KEY
        self.client = self.openai.Client()
        self.system_prompt = chat_system_content_message()
        self.model_id = 'ft:gpt-4o-2024-08-06:the-great-american-credit-secret-llc::BABaftly'

    async def get_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Extract 'user_query' and 'context'
            user_query = input_data.get("user_query")
            context = input_data.get("context", [])

            if not user_query:
                raise ValueError("Missing 'user_query' in input_data")

            print(f"[DEBUG] Sending question to OpenAI: {user_query} with context: {context}")

            # Prepare messages for OpenAI API
            messages = [{"role": "system", "content": self.system_prompt}]
            if context:
                messages.extend(context)  # Add context messages if available
            messages.append({"role": "user", "content": user_query})

            # Call OpenAI API
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract the bot response
            bot_response = response.choices[0].message.content.strip()
            print(f"[DEBUG] OpenAI Response: {bot_response}")

            # Return both user_query and bot_response
            return {"user_query": user_query, "bot_response": bot_response}

        except Exception as error:
            print(f"[DEBUG] Error in get_response: {error}")
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": "chat_service"})
            # Return user_query with None as bot_response in case of an error
            return {"user_id": input_data.get("user_id"),"user_query": input_data.get("user_query"), "bot_response": None}
        
async def credit_report_context_wrapper(state):
    controller = get_credit_report_controller()
    return await controller.get_credit_report_context(db, state["user_id"], state["user_query"])

async def fetch_previous_chat(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        previous_chat = await tool.fetch_previous_chat(state)
        print(f"[DEBUG] Fetched previous chat: {previous_chat}")
        return {"context": previous_chat if previous_chat else []}
    except Exception as e:
        logger.error(f"fetch_previous_chat failed: {e}")
        return {"context": []}  # Return an empty list if an error occurs

async def check_recent_data(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        controller = get_credit_report_controller()
        credit_context = await controller.get_credit_report_context(db, state["user_id"], state["user_query"])
        print(f"[DEBUG] Fetched report: {credit_context}")
        return {"context": credit_context if credit_context else []}
    except Exception as e:
        logger.error(f"get_credit_report_context failed: {e}")
        return {"context": []}  # Return an empty list if an error occurs

def setup_graph():
    print("[DEBUG] Setting up graph")  # Debug

    # Define the state with all required keys
    class State(TypedDict):
        user_id: str
        user_query: str
        is_verified: bool
        context: Annotated[list, operator.add]
        bot_response: str

    graph = StateGraph(State)
    tool = ToolFunction()
    chat_service = ChatService()

    # Step 1: Extract user query and user ID separately
    graph.add_node("get_user_query_and_id", tool.get_user_query_and_id)
    graph.add_node("get_user_query", tool.get_user_query)
    graph.add_node("get_user_id", tool.get_user_id)

    # Step 2: Check for credit report if verified
    graph.add_node("check_recent_data", check_recent_data)

    # Step 3: Fetch previous chat history
    graph.add_node("fetch_previous_chat", fetch_previous_chat)

    # Step 4: Merge the fetched context
    graph.add_node("merge_context", tool.merge_context)

    # Step 5: Process query with AI agent
    graph.add_node("agent1_process", chat_service.get_response)

    # Step 6: Store chat history
    async def store_chat_wrapper(state):
        return await tool.store_chat(state["user_id"], state["user_query"], state["bot_response"])

    graph.add_node("store_chat", store_chat_wrapper)

    # Conditional check for verification
    def check_verification(state):
        print("[DEBUG] Checking verification for state:", state)  # Debug
        if state.get("is_verified", False):
            return "check_recent_data"  # Fetch credit report only if verified
        return "fetch_previous_chat"  # Otherwise, fetch previous chat

    # Graph edges
    graph.add_edge(START, "get_user_query_and_id")
    graph.add_edge("get_user_query_and_id", "get_user_query")
    graph.add_edge("get_user_query_and_id", "get_user_id")
    graph.add_conditional_edges("get_user_query_and_id", check_verification)
    graph.add_edge("check_recent_data", "merge_context")
    graph.add_edge("fetch_previous_chat", "merge_context")
    graph.add_edge("merge_context", "agent1_process")  # Ensure agent1_process runs after merge_context
    graph.add_edge("agent1_process", "store_chat")  # Ensure store_chat runs after agent1_process

    print("[DEBUG] Graph setup completed")  # Debug
    return graph

async def main():
    graph = setup_graph()
    runnable_graph = graph.compile()
    test_input = {
        "user_id": "32b397c1-d160-44bc-9940-3d16542d8718",
        "user_query": "What is my credit score?",
        "is_verified": True
    }
    print(f"[DEBUG] Test input: {test_input}")  # Debug
    result = await runnable_graph.ainvoke(test_input)
    print("[DEBUG] Final Output:", result.get("bot_response"))  # Print the OpenAI response

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
