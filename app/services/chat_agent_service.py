import gc
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
from app.db import MONGO_URI
import operator

logger = setup_logger()

from pydantic import BaseModel
from app.repositories.chat_history_repositories import ChatHistoryRepository
from app.services.pinecone_vectorizer import CustomMetadataRetriever, OpenAIEmbedding, VectorizerEngine
from langgraph.graph import END, StateGraph
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from app.repositories.model_data_repositories import ModelDataRepository
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from app.repositories.credit_report_repositories import CreditReportRepository
from app.services.credit_report_extractor import CreditReportExtractor
from app.services.credit_report_processor import CreditReportProcessor
import pymongo


class UserQuery(BaseModel):
    user_id: str
    query: str

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
    mongo_client: any
    mongo_db: any
    vector_data: any
    pinecone_data_available: bool
    populate_vector_db: bool
    model_config: dict
    answer: str
    tools_initialized: bool




# --- Node1: Initialize tools ---
async def initialization_node(state):
    try:
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
        state["vectorizer"] = VectorizerEngine(
            encoder=encoder,
            vector_db_name=settings.VECTOR_DB_NAME,
            batch_size=10,
            dimension=settings.VECTOR_DIMENSION,
            namespace="credit_reports"
        )
        state["tools_initialized"] = True
        return state
    except Exception as error:
        print("initialization_node:: error - ",str(error))
        return state


# --- Node2: Load message history ---
async def load_history_node(state):
    try:
        mongo_history_repo = ChatHistoryRepository(state["user_id"], state["mongo_db"])
        state["chat_history"] = await mongo_history_repo.load_messages()
        state["mongo_history_repo"] = mongo_history_repo
        return state
    except Exception as error:
        print("load_history_node:: error - ",str(error))
        return state
    

# --- Node3: Check if user is verfied or not ---
async def check_for_verfied_condition(state):
    try:
        if state["is_verified"]:
            return "fetch_today_report_node"
        else:
            return "pull_model_config_node"
    except Exception as error:
        print("check_for_verfied_condition:: error - ",str(error))
        return "pull_model_config_node"

# --- Node4: Check if user report exists in mongo ---
async def fetch_today_report_node(state):
    try:
        data = await state["credit_report_repo"].get_todays_reoprt(state["mongo_db"], state["user_id"])
        state["current_credit_report"] = data
    except Exception as error:
        print("fetch_today_report_node:: error - ",str(error))
        return state


# --- Node4: Check if user report exists in mongo ---
async def check_today_report_condition(state):
    try:
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
        return "fetch_and_sync_new_data_node"
    

# ---Node5: Fetch and sync data explicitly async---
async def fetch_and_sync_new_data_node(state):
    try:
        state["pinecone_data_available"] = False
        isVectorized = state["current_credit_report"]["isVectorized"]
        report = state["current_credit_report"]
        if isVectorized:
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
        return state
    except Exception as error:
        print("fetch_and_sync_new_data_node:: error - ",str(error))
        return state



# --- Node6: Check if user data exists in Pinecone ---
async def fetch_vector_db_node(state):
    try:
        if not state["vectorizer"].vectordb:
            state["vectorizer"].load_vectorstore()
        res = state["vectorizer"].check_for_data(state["user_id"])
        if len(res['matches']) < 1:
            state["populate_vector_db"] = True
        else:
            state["populate_vector_db"] = False
        return state
    except Exception as error:
        print("check_vector_db_condition:: error - ",str(error))
        return state



async def check_vector_db_condition(state):
    try:
        if state["populate_vector_db"]:
            return "populate_vector_db_node"
        return "pull_model_config_node"
    except Exception as error:
        print("check_vector_db_condition:: error - ",str(error))
        return "pull_model_config_node"


# --- Node7: Populate Pinecone if missing ---
async def populate_vector_db_node(state):
    try:
        if not state["pinecone_data_available"]:
            vector_data = state["vector_data"]
            if not state["vectorizer"].vectordb:
                state["vectorizer"].load_vectorstore()
            await state["vectorizer"].create_vectorstore(vector_data, "report_data_id", "topics")
            state["pinecone_data_available"] = True
        return state
    except Exception as error:
        print("populate_vector_db_node:: error - ",str(error))
        return state


# --- Node8: Pull latest model configs ---
async def pull_model_config_node(state):
    try:
        models = await state["model_data_repo"].get_models(state["mongo_db"])
        state["model_config"] = models[0] if len(models) > 0 else {"model_id": settings.BASE_MODEL}
        return state
    except Exception as error:
        print("pull_model_config_node:: error - ",str(error))
        state["model_config"] = {"model_id": settings.BASE_MODEL}
        return state


# --- Node9: Conversational Retrieval generation ---
async def conversational_agent_node(state):
    try:
        memory = ConversationBufferMemory(
            chat_memory=ChatMessageHistory(messages=state["chat_history"]),
            return_messages=True,
            memory_key="chat_history"
        )

        llm = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY, temperature=0)

        retriever = CustomMetadataRetriever(
            user_id=state["user_id"], top_k=3
        ).as_langchain_retriever()

        chain_kwargs = {"llm": llm, "retriever": retriever, "memory": memory, "verbose": True}

        conversational_chain = ConversationalRetrievalChain.from_llm(**chain_kwargs)
        response = conversational_chain({"question": state["user_query"]})
        state["answer"] = response["answer"]
        return state
    except Exception as error:
        print("conversational_agent_node:: error - ",str(error))
        return state

# --- Node10: Persist message explicitly ---
async def persist_messages_node(state):
    try:
        state["mongo_history_repo"].add_user_message(state["user_query"])
        state["mongo_history_repo"].add_ai_message(state["answer"])
        return state
    except Exception as error:
        print("persist_messages_node:: error - ",str(error))
        return state



# --- Node11: Load message history ---
async def deinitialization_node(state):
    try:# Close MongoDB client explicitly
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
        return state
    except Exception as error:
        print("deinitialization_node:: error - ",str(error))
        return state


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
workflow.add_node("persist_messages_node", persist_messages_node)

# 🌟 NEW explicitly added deinitialization node
workflow.add_node("deinitialization_node", deinitialization_node)

# Edge Setup explicitly clear and adjusted
workflow.set_entry_point("initialization_node")
workflow.add_edge("initialization_node","load_history_node")

workflow.add_conditional_edges("load_history_node", check_for_verfied_condition)

workflow.add_conditional_edges("fetch_today_report_node",check_today_report_condition,)

workflow.add_edge("fetch_and_sync_new_data_node","fetch_vector_db_node")


workflow.add_conditional_edges("fetch_vector_db_node",check_vector_db_condition)

workflow.add_edge("populate_vector_db_node", "pull_model_config_node")
workflow.add_edge("pull_model_config_node","conversational_agent_node")
workflow.add_edge("conversational_agent_node","persist_messages_node")

# Explicit Edge clearly set to deinitialize after persist node
workflow.add_edge("persist_messages_node","deinitialization_node")
workflow.add_edge("deinitialization_node", END)


app_graph = workflow.compile()


print(app_graph.get_graph().print_ascii())