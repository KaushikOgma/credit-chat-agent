
import datetime
import traceback
from typing import List, Union

from pymongo import MongoClient
import pymongo
from app.utils.config import settings
from app.utils.constants import DBCollections
from app.utils.helpers.common_helper import generate_uuid
from app.utils.helpers.password_helper import hash_password
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.schemas.metadata_schema import MetadataSchema
from app.utils.helpers.auth_helper import generate_api_key
from app.schemas.model_data_schema import ModelDataSchema
from app.attribute_selector.credit_report_attributes import CreditReportProjections
from pymongo.database import Database
from langchain.schema import HumanMessage, AIMessage
from app.utils.constants import DBCollections
from collections import deque
from app.db import MONGO_URI
from app.utils.logger import setup_logger
logger = setup_logger()


class ChatHistoryRepository:
    def __init__(self, user_id: str, db: any):
        self.user_id = user_id
        self.db = db
        self.collection = self.db[DBCollections.CHAT_HISTORY.value]

    async def load_messages(self):
        docs = self.collection.find({"user_id": self.user_id}).sort("timestamp", pymongo.ASCENDING)
        messages = []
        question_count = 0
        for doc in docs:
            if doc["sender"] == "human":
                question_count += 1
                messages.append(HumanMessage(content=doc["content"]))
            else:
                messages.append(AIMessage(content=doc["content"]))
        messages = deque(messages, maxlen=settings.CHAT_HISTORY_LIMIT*2) 
        messages = list(messages)
        return messages, question_count

    async def add_user_message(self, content: str, question_number: int):
        self.collection.insert_one({
            "user_id": self.user_id, 
            "sender": "human",
            "content": content,
            "question_number": question_number,
            "timestamp": datetime.datetime.now(datetime.UTC)
        })

    async def add_ai_message(self, content: str, question_number: int):
        self.collection.insert_one({
            "user_id": self.user_id, 
            "sender": "ai",
            "content": content,
            "question_number": question_number,
            "timestamp": datetime.datetime.now(datetime.UTC)
        })

    async def clear(self):
        self.collection.delete_many({"user_id": self.user_id})

    