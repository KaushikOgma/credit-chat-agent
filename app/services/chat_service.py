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
from app.db import get_db_instance
import operator

logger = setup_logger()

db = get_db_instance()

class ToolFunction:
    def __init__(self):
        self.chat_limit = settings.CHAT_HISTORY_LIMIT
        # Use a different database name for chat storage
        self.chat_db = db  # Synchronous database instance
        self.chat_collection = self.chat_db["chat_history"]

    async def fetch_previous_chat(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[DEBUG] Input to fetch_previous_chat: {state}")  # Debugging
        user_id = state.get("user_id")
        print(user_id)  # Debugging
        if not user_id:
            raise ValueError("Missing user_id in fetch_previous_chat")
        
        # Run the synchronous `find_one` in a thread
        chat_data = await asyncio.to_thread(self.chat_collection.find_one, {"user_id": user_id})
        chat_history = chat_data.get("chat_history", [])[-self.chat_limit:] if chat_data else []
        print(f"[DEBUG] Output from fetch_previous_chat")
        # Return the chat history wrapped in a dictionary
        # state["chat_history"] = chat_history
        # return state
        return {"chat_history": chat_history, "context": state["context"] if "context" in state else []}  # Return the chat history directly
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
            # Run the synchronous `update_one` in a thread
            await asyncio.to_thread(
                self.chat_collection.update_one,
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

        context = []
        chat_history = []
        if "chat_history" in data:
            chat_ref = ""
            for item in data["chat_history"]:
                chat_ref += f"USER: {item['user_query']}\n ASSISTANT: {item['bot_response']}\n\n"
            if len(chat_ref) > 1:
                chat_history.append(chat_ref)
        if "context" in data:
            curr_context = ""
            for item in data["context"]:
                curr_context += f"{item['context']}\n\n"
            if len(curr_context) > 1:
                print(">>>>>>>>>>>>>>>>> curr_context:: ",curr_context)
                context.append(curr_context)
                print(">>>>>>>>>>>>>>>>> context:: ",context)
        merged_data = {"context": context, "chat_history": chat_history}
        print(f"[DEBUG] Output from merge_context: {merged_data}")
        # return  {"context": context, "chat_history": chat_history}
        return {"merged_data": merged_data} 

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
    
    async def get_contextual_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        merged_data = input_data.get("merged_data",{})
        user_query = input_data.get("user_query")
        context = merged_data.get("context",input_data.get("context", None))
        context = context[0] if context is not None else None
        chat_history = merged_data.get("chat_history",input_data.get("chat_history", None))
        chat_history = chat_history[0] if chat_history is not None else None
        user_id = input_data.get("user_id")

        # Validate input data
        if not user_query:
            raise ValueError("Missing 'user_query' in input_data")
        

        # Format the question string with actual values
        question = ""
        if chat_history is not None:
            question += f"""
            The conversation referance:
            ------------------------
            {chat_history}
        
            """
        if context is not None:
            question += f"""
            Additional context:
            ------------------------
            {context}
        
            """
        question += f"""
        User's question:
        ------------------------
        {user_query}
    
        Note:
        -------
        Given the conversation history and the additional context, provide the best possible answer.
        If you are unsure about some details, you may provide disclaimers or clarifications.
        """
        print(f"[DEBUG] Sending question to OpenAI: {question}")

        try:
            # Call OpenAI API asynchronously
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract the bot response
            bot_response = response.choices[0].message.content.strip()
            print(f"[DEBUG] OpenAI Response: {bot_response}")

            # Return both user_query and bot_response
            return {"user_id": user_id, "user_query": user_query, "bot_response": bot_response}

        except Exception as error:
            # Log the error and return a fallback response
            print(f"[DEBUG] Error in get_contextual_response: {error}")
            logger.exception(
                "Error in get_contextual_response",
                extra={"moduleName": settings.MODULE, "serviceName": "chat_service"}
            )
            return {
                "user_id": input_data.get("user_id"),
                "user_query": user_query,
                "bot_response": None
            }
        
async def credit_report_context_wrapper(state):
    controller = get_credit_report_controller()
    return await controller.get_credit_report_context(db, state["user_id"], state["user_query"])

async def check_recent_data(state: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[DEBUG] Input to check_recent_data: {state}")  # Debug
    try:
        controller = get_credit_report_controller()
        credit_context = await controller.get_credit_report_context(db, state["user_id"], state["user_query"])
        # print(f"[DEBUG] Fetched report: {credit_context}")
        if isinstance(credit_context, str):
            print("[DEBUG] Converting string context to list")
            credit_context = [{"context": credit_context}]
        # state["context"] = credit_context if credit_context else []
        # return state
        return {"context": credit_context if credit_context else [], "chat_history": state["chat_history"] if "chat_history" in state else []}  # Return the context directly
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
        chat_history: list
        merged_data: dict

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
    graph.add_node("fetch_previous_chat", tool.fetch_previous_chat)

    # Step 4: Merge the fetched context
    graph.add_node("merge_context", tool.merge_context)

    # Step 5: Process query with AI agent
    graph.add_node("agent1_process", chat_service.get_contextual_response)

    # Step 6: Store chat history
    async def store_chat_wrapper(state):
        return await tool.store_chat(state["user_id"], state["user_query"], state["bot_response"])

    graph.add_node("store_chat", store_chat_wrapper)

    # Conditional check for verification
    def check_verification(state):
        print("[DEBUG] Checking verification for state:", state)  # Debug
        if state.get("is_verified", False):
            # Trigger both `check_recent_data` and `fetch_previous_chat`
            return "trigger_check_recent_data"
        # Trigger only `fetch_previous_chat` if not verified
        return "trigger_fetch_previous_chat"

    # Graph edges
    graph.add_edge(START, "get_user_query_and_id")
    graph.add_edge("get_user_query_and_id", "get_user_query")
    graph.add_edge("get_user_query_and_id", "get_user_id")
    graph.add_conditional_edges("get_user_query_and_id", check_verification,{
        "trigger_check_recent_data": "check_recent_data",
        "trigger_fetch_previous_chat": "fetch_previous_chat"
    })
    graph.add_edge("check_recent_data", "fetch_previous_chat")
    graph.add_edge("fetch_previous_chat", "merge_context")
    graph.add_edge("merge_context", "agent1_process")  # Ensure agent1_process runs after merge_context
    graph.add_edge("agent1_process", "store_chat")  # Ensure store_chat runs after agent1_process

    print("[DEBUG] Graph setup completed")  # Debug
    return graph

async def main():
    graph = setup_graph()
    runnable_graph = graph.compile()
    print(runnable_graph.get_graph().print_ascii())
 
    test_input = {
        "user_id": "32b397c1-d160-44bc-9940-3d16542d8718",
        "user_query": "What is my credit score?",
        "is_verified": True,
        "context": [],
        "chat_history": [],
        "merged_data": {}
        
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
