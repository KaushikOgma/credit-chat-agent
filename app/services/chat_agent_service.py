import gc
import os
import sys
import traceback
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
sys.path.append(os.getcwd())
import asyncio
from typing import List, Union, Dict, Any, Annotated, TypedDict
from app.utils.helpers.prompt_helper import chat_system_content_message, condense_question_system_content_message
from app.utils.config import settings
from app.utils.logger import setup_logger
from setuptools._distutils.util import strtobool
from app.db import MONGO_URI
from pydantic import BaseModel
from app.repositories.chat_history_repositories import ChatHistoryRepository
from app.services.pinecone_vectorizer import OpenAIEmbedding, VectorizerEngine
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts.prompt import PromptTemplate
from app.repositories.model_data_repositories import ModelDataRepository
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from app.repositories.credit_report_repositories import CreditReportRepository
from app.services.credit_report_extractor import CreditReportExtractor
from app.services.credit_report_processor import CreditReportProcessor
import pymongo

logger = setup_logger()


class State(dict):
    user_id: str
    user_query: str
    is_verified: bool
    is_premium: bool
    chat_history: list
    credit_report_processor_service: CreditReportProcessor
    credit_report_extractor_service: CreditReportExtractor
    mongo_history_repo: ChatHistoryRepository
    credit_report_repo: CreditReportRepository
    model_data_repo: ModelDataRepository
    encoder: OpenAIEmbedding
    vectorizer: VectorizerEngine
    current_credit_report: dict
    chain_kwargs: dict
    mongo_client: any
    mongo_db: any
    vector_data: any
    pinecone_data_available: bool
    populate_vector_db: bool
    model_config: dict
    question_number: int
    answer: str
    tools_initialized: bool
    path: list
    error_occured: bool
    error_details: dict
    next_node: str
    non_verified_response: bool


# --- Node1: Initialize tools ---
async def initialization_node(state):
    """Initialize the tools and services needed for the workflow.
    This function sets up the MongoDB connection, initializes the credit report processor and extractor services,

    Args:
        state (State): The current state of the workflow.

    Returns:
        State: The updated state with initialized tools and services.
    """    
    state["error_occured"] = False 
    state["non_verified_response"] = False
    state["next_node"] = "load_history_node"
    try:
        print("initialization_node:: ")
        # Explicit Mongo Connection
        mongo_client = pymongo.MongoClient(MONGO_URI)
        state["mongo_client"] = mongo_client
        state["mongo_db"] = mongo_client[settings.DB_NAME]
        state["credit_report_processor_service"] = CreditReportProcessor()
        state["credit_report_extractor_service"] = CreditReportExtractor()
        state["credit_report_repo"] = CreditReportRepository() 
        state["model_data_repo"] = ModelDataRepository() 
        encoder = OpenAIEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)
        state["encoder"] = encoder
        state["chain_kwargs"]= {"verbose": bool(strtobool(settings.VERBOSE))}
        state["vectorizer"] = VectorizerEngine(
            encoder=encoder,
            vector_db_name=settings.VECTOR_DB_NAME,
            batch_size=10,
            dimension=settings.VECTOR_DIMENSION,
            namespace="credit_reports"
        )
        state["tools_initialized"] = True
        state["path"] = ["initialization_node"]
        return state
    except Exception as error:
        print("initialization_node:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "initialization_node"
        }
        return state


# --- Node2: Load message history ---
async def load_history_node(state):
    """Load the message history from MongoDB and set up the chat history for the user.
    This function is called at the beginning of the workflow to retrieve the user's previous messages.

    Args:
        state (State): The current state of the workflow.

    Returns:
        State: The updated state with the loaded chat history and question number.
    """
    state["next_node"] = "load_history_node"
    try:
        print("load_history_node:: ")
        mongo_history_repo = ChatHistoryRepository(state["user_id"], state["mongo_db"])
        chat_history, question_count = await mongo_history_repo.load_messages()
        state["question_number"] = question_count + 1
        state["chat_history"] = chat_history
        state["mongo_history_repo"] = mongo_history_repo
        state["chain_kwargs"]["memory"] = ConversationBufferMemory(
            chat_memory=ChatMessageHistory(messages=chat_history),
            return_messages=True,
            memory_key="chat_history"
        )
        state["path"].append("load_history_node")
        return state
    except Exception as error:
        print("load_history_node:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "load_history_node"
        }
        return state
    

# --- cond edge 1: Check if user is verfied or not ---
async def check_for_verfied_condition(state):
    """Checks if the user is verified or not.
    This is used to determine the next node in the workflow.
    
    Args:
        state (State): The current state of the workflow.

    Returns:
        str: The next node to be executed in the workflow.
    """
    try:
        print("check_for_verfied_condition:: ")
        if state["error_occured"]:
            return "error_handler_node"
        else:
            if state["is_verified"]:
                return "fetch_today_report_node"
            else:
                return "pull_model_config_node"
    except Exception as error:
        print("check_for_verfied_condition:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "check_for_verfied_condition"
        }
        return "error_handler_node"

# --- Node3: fetch user's report for today from mongo db ---
async def fetch_today_report_node(state):
    """Fetch the user's credit report for today from MongoDB.
    This function retrieves the report and checks if it has been vectorized.
    If the report is not found, it will be handled in the next node.
    
    Args:
        state (State): The current state of the workflow.

    Returns:
        State: The updated state with the fetched credit report and vectorization status.
    """    
    try:
        print("fetch_today_report_node:: ")
        data = await state["credit_report_repo"].get_todays_reoprt(state["mongo_db"], state["user_id"])
        state["current_credit_report"] = data
        if data is not None:
            state["pinecone_data_available"] = data["isVectorized"]
        state["path"].append("fetch_today_report_node")
        return state
    except Exception as error:
        print("fetch_today_report_node:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "fetch_today_report_node"
        }
        return state


# --- cond edge 2: Check if user's report for today exists in mongo ---
async def check_today_report_condition(state):
    """Check if the user's credit report for today exists in MongoDB.
    This function determines the next node in the workflow based on the report's vectorization status.
    If the report is not found, it will be handled in the next node.
    
    Args:
        state (State): The current state of the workflow.

    Returns:
        str: The next node to be executed in the workflow.
    """    
    try:
        print("fetch_today_report_node:: ")
        if state["error_occured"]:
            return "error_handler_node"
        else:
            data = state["current_credit_report"]
            if data:
                if data["isVectorized"]:
                    return "fetch_vector_db_node"
                else:
                    return "fetch_and_sync_new_data_node"
            else:
                return "fetch_and_sync_new_data_node"
    except Exception as error:
        print("check_today_report_condition:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "check_today_report_condition"
        }
        return "error_handler_node"
    

# ---Node4: Fetch and sync data explicitly ---
async def fetch_and_sync_new_data_node(state):
    """Fetch and sync new data from the user's credit report.
    This function processes the report, vectorizes it, and updates the MongoDB and Pinecone database accordingly.
    
    Args:
        state (State): The current state of the workflow.

    Returns:
        State: The updated state with the processed credit report and vectorization status.
    """    
    state["next_node"] = "fetch_vector_db_node"
    try:
        print("fetch_and_sync_new_data_node:: ")
        state["pinecone_data_available"] = False
        report = state["current_credit_report"]
        if report is not None:
            isVectorized = state["current_credit_report"]["isVectorized"]
            if not isVectorized:
                mongo_data, vector_data = await state["credit_report_processor_service"].process_report(credit_report_json=None, user_id=state["user_id"], categorized_resp=report["report"])   
                state["vector_data"] = vector_data
                print("credit report processed")
                # Sync the vector DB with the latest QA pairs
                if not state["vectorizer"].vectordb:
                    state["vectorizer"].load_vectorstore()
                await state["vectorizer"].create_vectorstore(vector_data, "report_data_id", "topics")
                print("credit report added to vector db")
                await state["credit_report_repo"].update_report(state["mongo_db"], report["_id"], {"isVectorized": True})
            state["pinecone_data_available"] = True
        else:
            # There are no report in the mongo db for today
            credit_report = await state["credit_report_extractor_service"].get_credit_report(state["user_id"])     
            if credit_report:
                print("credit report found")
                mongo_data, vector_data = await state["credit_report_processor_service"].process_report(user_id=state["user_id"], credit_report_json=credit_report, categorized_resp=None)   
                state["vector_data"] = vector_data
                if mongo_data is not None:
                    print("credit report processed")
                    mongo_data["isVectorized"] = False
                    inserted_id = await state["credit_report_repo"].add_report(state["mongo_db"], mongo_data)
                    print("credit report added:: ",inserted_id)
                    if not state["vectorizer"].vectordb:
                        state["vectorizer"].load_vectorstore()
                    await state["vectorizer"].create_vectorstore(vector_data, "report_data_id", "topics")
                    print("credit report added to vector db")
                    await state["credit_report_repo"].update_report(state["mongo_db"], inserted_id, {"isVectorized": True})
                    print("Make credit report is vectorize true")
                    mongo_data["isVectorized"] = True
                    state["current_credit_report"] = mongo_data
                    state["pinecone_data_available"] = True
        state["path"].append("fetch_and_sync_new_data_node")
        return state
    except Exception as error:
        print("fetch_and_sync_new_data_node:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "fetch_and_sync_new_data_node"
        }
        return state



# --- Node5: fetch user report from vector db ---
async def fetch_vector_db_node(state):
    """Fetch the vector database node and check if the user data exists in Pinecone.
    This function is used to determine if the user data needs to be populated in the vector database.
    
    Args:
        state (State): The current state of the workflow.

    Returns:
        State: The updated state with the vector database status.
    """    
    try:
        print("fetch_vector_db_node:: ")
        if not state["vectorizer"].vectordb:
            state["vectorizer"].load_vectorstore()
        res = await state["vectorizer"].check_for_data(state["user_id"])
        if len(res['matches']) < 1:
            state["populate_vector_db"] = True
        else:
            state["populate_vector_db"] = False
        state["path"].append("fetch_vector_db_node")
        return state
    except Exception as error:
        print("fetch_vector_db_node:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "fetch_vector_db_node"
        }
        return state



# --- cond edge 3: Check if user data exists in Pinecone ---
async def check_vector_db_condition(state):
    """Check if the vector database needs to be populated or not.
    This function is used to determine the next node in the workflow based on the vector database status.
    
    Args:
        state (State): The current state of the workflow.

    Returns:
        str: The next node to be executed in the workflow.
    1. If the vector database needs to be populated, return "populate_vector_db_node".
    2. Otherwise, return "pull_model_config_node".
    """    
    try:
        print("check_vector_db_condition:: ")
        if state["error_occured"]:
            return "error_handler_node"
        else:
            if state["populate_vector_db"]:
                return "populate_vector_db_node"
            else:
                return "pull_model_config_node"
    except Exception as error:
        print("check_vector_db_condition:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "check_vector_db_condition"
        }
        return "error_handler_node"


# --- Node6: Populate Pinecone if missing ---
async def populate_vector_db_node(state):
    """Populate the vector database with the user's data if it is missing.
    This function is called if the vector database needs to be populated.
    
    Args:
        state (State): The current state of the workflow.

    Returns:
        State: The updated state with the populated vector database.
    1. If the vector database is not available, load it and create a new vector store.
    2. Set the "pinecone_data_available" flag to True.
    3. Append the current node to the state path.
    """    
    state["next_node"] = "pull_model_config_node"
    try:
        print("populate_vector_db_node:: ")
        if not state["pinecone_data_available"]:
            vector_data = state["vector_data"]
            if not state["vectorizer"].vectordb:
                state["vectorizer"].load_vectorstore()
            await state["vectorizer"].create_vectorstore(vector_data, "report_data_id", "topics")
            state["pinecone_data_available"] = True
        state["path"].append("populate_vector_db_node")
        return state
    except Exception as error:
        print("populate_vector_db_node:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "populate_vector_db_node"
        }
        return state


# --- Node7: Pull latest model configs ---
async def pull_model_config_node(state):
    """Pull the latest model configuration from the database.
    This function retrieves the model configuration and sets it in the state.
    
    Args:
        state (State): The current state of the workflow.

    Returns:
        State: The updated state with the model configuration.
    1. If the model configuration is not found, set the default model ID.
    2. Append the current node to the state path.
    3. Return the updated state.
    4. If an error occurs, set the default model ID and return the state.
    5. Print the error message and the traversed path.
    6. Return the state with the default model ID.
    """    
    state["next_node"] = "conversational_agent_node"
    try:
        print("pull_model_config_node:: ")
        models = await state["model_data_repo"].get_models(state["mongo_db"])
        state["model_config"] = models[0] if len(models) > 0 else {"model_id": settings.BASE_MODEL}
        state["path"].append("pull_model_config_node")
        return state
    except Exception as error:
        print("pull_model_config_node:: error - ",str(error))
        state["model_config"] = {"model_id": settings.BASE_MODEL}
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "pull_model_config_node"
        }
        return state


# --- Node8: Conversational Retrieval generation ---
async def conversational_agent_node(state):
    """Generate a conversational response using the LangChain ConversationalRetrievalChain.
    This function sets up the language model, retriever, and prompt templates for the conversation.

    Args:
        state (State): The current state of the workflow.
        question (str): The user's question.
        context (str): The context for the conversation.
        chat_system_content (str): The system content for the chat.

    Returns:
        State: The updated state with the generated answer.
    1. Initialize the language model and retriever based on the user's verification status.
    2. Set up the prompt templates for the conversation.
    3. Create the ConversationalRetrievalChain and generate the response.
    4. Append the current node to the state path.
    5. Return the updated state.
    6. If an error occurs, print the error message and the traversed path.
    7. Return the state with the error message.
    8. Print the error message and the traversed path.
    9. Return the state with the error message.
    10. Print the error message and the traversed path.
    """    
    state["next_node"] = "check_non_verified_response_node"
    try:
        print("conversational_agent_node:: ")
        llm = ChatOpenAI(model= state["model_config"]["model_id"],openai_api_key=settings.OPENAI_API_KEY, temperature=0)
        state["chain_kwargs"]["llm"] = llm

        if state["is_verified"]:
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index = pc.Index(settings.VECTOR_DB_NAME)
            embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY, model=settings.EMBEDDING_MODEL_NAME)
            vectorstore = PineconeVectorStore(
                    index=index,
                    embedding=embeddings,
                    distance_strategy="COSINE",
                    namespace="credit_reports",
                    text_key="summary"
                )
            filter_dict = {"$and": [{"userId": {"$in": [state["user_id"]]}}]}
            retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 3, 'filter': filter_dict})
            state["chain_kwargs"]["retriever"] = retriever
        else:
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index = pc.Index(settings.VECTOR_DB_NAME)
            embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY, model=settings.EMBEDDING_MODEL_NAME)
            vectorstore = PineconeVectorStore(
                    index=index,
                    embedding=embeddings,
                    distance_strategy="COSINE",
                    text_key="summary"
                )
            filter_dict = {"$and": [{"userId": {"$in": [state["user_id"]]}}]}
            retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 3, 'filter': filter_dict})
            state["chain_kwargs"]["retriever"] = retriever

        chat_system_template = f"""
        {chat_system_content_message()}
        """ + """
        CONTEXT: 
        {context}
        QUESTION:  {question}
        """
        # # Create the prompt templates:
        # chat_messages = [
        #     SystemMessagePromptTemplate.from_template(chat_system_template),
        #     HumanMessagePromptTemplate.from_template("{question}"),
        # ]
        # chat_prompt = ChatPromptTemplate.from_messages(chat_messages)
        chat_prompt = PromptTemplate.from_template(chat_system_template)
        state["chain_kwargs"]["combine_docs_chain_kwargs"] = {"prompt": chat_prompt}


        condense_question_prompt = PromptTemplate.from_template(condense_question_system_content_message())
        state["chain_kwargs"]["condense_question_prompt"] = condense_question_prompt


        conversational_chain = ConversationalRetrievalChain.from_llm(**state["chain_kwargs"])
        response = conversational_chain.invoke({"question": state["user_query"]})
        state["answer"] = response["answer"]
        state["path"].append("conversational_agent_node")
        return state
    except Exception as error:
        print("conversational_agent_node:: error - ",str(error))
        print(traceback.format_exc())
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "conversational_agent_node"
        }
        return state



# --- Node9: check for non verified token in answer ---
async def check_non_verified_response_node(state, token="**<<NA>>**"):
    state["next_node"] = "persist_messages_node"
    try:
        response = state["answer"]
        # Strip any whitespace from the end of the response
        response = response.strip()
        # Check if the response ends with the specified token
        if response.endswith(token):
            state["non_verified_response"] = True
            state["answer"] = response[:-len(token)].rstrip()
        else:
            state["non_verified_response"] = False
        return state
    except Exception as error:
        print("check_non_verified_response_node:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "check_non_verified_response_node"
        }
        return state

    

# --- Node10: Persist message explicitly ---
async def persist_messages_node(state):
    """Persist the user and AI messages in the MongoDB database.
    This function is called after the conversational agent generates a response.

    Args:
        state (State): The current state of the workflow.
        user_query (str): The user's question.
        answer (str): The AI-generated answer.
        question_number (int): The question number for the conversation.

    Returns:
        State: The updated state with the persisted messages.
    1. Add the user message to the MongoDB database.
    """    
    state["next_node"] = "deinitialization_node"
    try:
        print("persist_messages_node:: ")
        await state["mongo_history_repo"].add_user_message(state["user_query"], state["question_number"])
        await state["mongo_history_repo"].add_ai_message(state["answer"], state["question_number"])
        state["path"].append("persist_messages_node")
        return state
    except Exception as error:
        print("persist_messages_node:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "persist_messages_node"
        }
        return state



# --- Node11: Load message history ---
async def deinitialization_node(state):
    """Deinitialize the tools and services used in the workflow.
    This function closes the MongoDB connection, deinitializes the encoder and vectorizer connections,
    and cleans up any extra services.
    It also cleans up the state flags and performs garbage collection.
    
        Finally, it appends the current node to the state path.

    Args:
        state (State): The current state of the workflow.

    Returns:
        State: The updated state after deinitialization.
    1. Close the MongoDB client explicitly.
    2. Deinitialize encoder connections (if needed explicitly).
    3. Deinitialize vectorizer connections (if applicable explicitly).
    4. Deinitialize any extra services explicitly (if needed).
    5. Clean state flags explicitly.
    6. Perform garbage collection explicitly.
    7. Append the current node to the state path.
    8. Return the updated state.
    9. If an error occurs, print the error message and the traversed path.
    """    
    try:# Close MongoDB client explicitly
        print("deinitialization_node:: ")
        mongo_client = state.get("mongo_client", None)
        if mongo_client:
            mongo_client.close()
            state["mongo_db"] = None
            state["mongo_client"] = None

        # Deinitialize encoder connections (if needed explicitly)
        encoder = state.get("encoder", None)
        if encoder:
            state["encoder"] = None

        # Deinitialize vectorizer connections (if applicable explicitly)
        vectorizer = state.get("vectorizer", None)
        if vectorizer:
            vectorizer.unload_vectorstore()

        # Deinitialize any extra services explicitly (if needed)
        processor_service = state.get("credit_report_processor_service", None)
        if processor_service:
            state["credit_report_processor_service"] = None


        extractor_service = state.get("credit_report_extractor_service", None)
        if extractor_service:
            state["credit_report_extractor_service"] = None

        credit_report_repo = state.get("credit_report_repo", None)
        if credit_report_repo:
            state["credit_report_repo"] = None

        model_data_repo = state.get("model_data_repo", None)
        if model_data_repo:
            state["model_data_repo"] = None

        # Clean state flags explicitly
        state["tools_initialized"] = False
        await asyncio.to_thread(gc.collect)
        state["path"].append("deinitialization_node")
        return state
    except Exception as error:
        print("deinitialization_node:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "deinitialization_node"
        }
        return state



# --- cond edge 4: error handle condition explicitly ---
async def check_error_condition(state): 
    """Check if an error has occurred in the workflow.
    This function is used to determine the next node in the workflow based on the error status.
 
    Args:
        state (State): The current state of the workflow.
 
    Returns:
        str: The next node to be executed in the workflow.
    1. If an error has occurred, return "error_handler_node".
    2. Otherwise, return the next node in the workflow.
    3. If an error occurs, print the error message and the traversed path.
    4. Capture the traceback string and store it in the state for later handling or processing.
    5. Store the error message and traceback in the state for later handling or processing.
    6. Set the "error_occured" flag to True.
    7. Store the error details in the state.
    8. Return "error_handler_node".
    """
    try:
        print("check_error_condition:: ")
        if state["error_occured"]:
            return "error_handler_node"
        else:
            return state["next_node"]
    except Exception as error:
        print("check_error_condition:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "check_error_condition"
        }
        return "error_handler_node"


# --- Node12: error handle explicitly ---
async def error_handler_node(state):
    """Handle errors that occur during the workflow.
    This function captures the error details and appends them to the state.
    It also prints the error message and the traversed path for debugging purposes.
 
    Args:
        state (State): The current state of the workflow.
 
    Returns:
        State: The updated state with error details.
    1. Print the error message and the traversed path.
    2. Capture the traceback string and store it in the state.
    3. Store the error message and traceback in the state for later handling or processing.
    4. Set the "error_occured" flag to True.
    5. Return the updated state.
    """  
    try:
        print("error_handler_node:: ")
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        print("error details:: ", state["error_details"])
        return state
    except Exception as error:
        print("error_handler_node:: error - ",str(error))
        print("Travarsed Path:: ", " --> ".join(elm for elm in state.get("path",[])))
        # Capture traceback string
        tb_str = traceback.format_exception(error)
        # Store error message and traceback in state for later handling or processing
        state["error_occured"] = True
        state["error_details"] = {
            "message": str(error),
            "traceback": ''.join(tb_str).strip(),
            "node": "error_handler_node"
        }
        return state


async def build_state_graph():
    """Build the state graph for the LangGraph workflow.
    This function defines the nodes and edges of the workflow, including the initialization, data fetching,
    vectorization, and conversational agent nodes.
    It also sets up the conditional edges based on the user's verification status and the vector database status.

    Returns:
        StateGraph: The compiled state graph for the workflow.
    1. Define the nodes for the workflow.
    2. Define the edges for the workflow.
    3. Set the entry point for the workflow.
    4. Compile the workflow into a runnable graph.
    5. Return the runnable graph.
    6. If an error occurs, print the error message and the traversed path.
    """    
    try:
        # --- LangGraph workflow explicitly defined ---
        workflow = StateGraph(State)


        # Existing Nodes explicitly added
        workflow.add_node("initialization_node", initialization_node)
        workflow.add_node("load_history_node", load_history_node)
        workflow.add_node("fetch_today_report_node", fetch_today_report_node)
        workflow.add_node("fetch_and_sync_new_data_node", fetch_and_sync_new_data_node)
        workflow.add_node("fetch_vector_db_node", fetch_vector_db_node)
        workflow.add_node("populate_vector_db_node", populate_vector_db_node)
        workflow.add_node("pull_model_config_node", pull_model_config_node)
        workflow.add_node("conversational_agent_node", conversational_agent_node)
        workflow.add_node("check_non_verified_response_node", check_non_verified_response_node)
        workflow.add_node("persist_messages_node", persist_messages_node)
        workflow.add_node("deinitialization_node", deinitialization_node)
        workflow.add_node("error_handler_node", error_handler_node) 


        # Edge Setup explicitly clear and adjusted
        workflow.set_entry_point("initialization_node")
        workflow.add_conditional_edges("initialization_node", check_error_condition, {
            "error_handler_node": "error_handler_node",
            "load_history_node": "load_history_node"
        })
        workflow.add_conditional_edges("load_history_node", check_for_verfied_condition,{
            "error_handler_node": "error_handler_node",
            "fetch_today_report_node": "fetch_today_report_node",
            "pull_model_config_node": "pull_model_config_node"
        })
        workflow.add_conditional_edges("fetch_today_report_node",check_today_report_condition,{
            "error_handler_node": "error_handler_node",
            "fetch_vector_db_node": "fetch_vector_db_node",
            "fetch_and_sync_new_data_node": "fetch_and_sync_new_data_node"
        })
        workflow.add_conditional_edges("fetch_and_sync_new_data_node", check_error_condition, {
            "error_handler_node": "error_handler_node",
            "fetch_vector_db_node": "fetch_vector_db_node"
        })
        workflow.add_conditional_edges("fetch_vector_db_node",check_vector_db_condition,{
            "error_handler_node": "error_handler_node",
            "populate_vector_db_node": "populate_vector_db_node",
            "pull_model_config_node": "pull_model_config_node"
        })
        workflow.add_conditional_edges("populate_vector_db_node", check_error_condition, {
            "error_handler_node": "error_handler_node",
            "pull_model_config_node": "pull_model_config_node"
        })
        workflow.add_conditional_edges("pull_model_config_node", check_error_condition, {
            "error_handler_node": "error_handler_node",
            "conversational_agent_node": "conversational_agent_node"
        })
        workflow.add_conditional_edges("conversational_agent_node", check_error_condition, {
            "error_handler_node": "error_handler_node",
            "check_non_verified_response_node": "check_non_verified_response_node"
        })
        workflow.add_conditional_edges("check_non_verified_response_node", check_error_condition, {
            "error_handler_node": "error_handler_node",
            "persist_messages_node": "persist_messages_node"
        })
        workflow.add_conditional_edges("persist_messages_node", check_error_condition, {
            "error_handler_node": "error_handler_node",
            "deinitialization_node": "deinitialization_node"
        })
        workflow.add_edge("error_handler_node", "deinitialization_node")
        workflow.add_edge("deinitialization_node", END)
        runnable_graph = workflow.compile()
        return runnable_graph
    except Exception as error:
        print("build_state_graph:: error - ",str(error))
        raise error


runnable_graph = asyncio.run(build_state_graph())
# print(runnable_graph.get_graph().print_ascii())
# print(runnable_graph.get_graph().draw_mermaid())

async def start():
    test_input = {
        "user_id": "32b397c1-d160-44bc-9940-3d16542d8718",
        "user_query": "what is chatGPT?",
        "is_verified": True,
        "is_premium": False
    }
    print(f"[DEBUG] Test input: {test_input}")  # Debug
    result = await runnable_graph.ainvoke(test_input)
    print("[DEBUG] Final Output:", result.get("answer"))  # Print the OpenAI response
    print("Travarsed Path:: ", " --> ".join(elm for elm in result.get("path",[])))

if __name__ == "__main__":
    asyncio.run(start())