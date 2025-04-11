
import datetime
import traceback
from typing import List, Union

from bson import ObjectId
from pymongo import MongoClient
import pymongo
from app.attribute_selector.chat_attributes import ChatProjections
from app.utils.config import settings
from app.utils.constants import DBCollections
from app.utils.helpers.common_helper import generate_uuid
from app.utils.helpers.password_helper import hash_password
from app.utils.helpers.date_helper import get_user_time, convert_timezone, get_utc_time
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
    def __init__(self, user_id: str, credit_service_user_id: str, db: any, input_timezone = None):
        self.user_id = user_id
        self.credit_service_user_id = credit_service_user_id
        self.db = db
        self.input_timezone = input_timezone if input_timezone is not None else settings.APP_TIMEZONE
        self.collection = self.db[DBCollections.CHAT_HISTORY.value]
        self.service_name = "chat_service"

    async def load_messages(self):
        messages = []
        question_count = 0
        try:
            docs = self.collection.find({"user_id": self.user_id}).sort("timestamp", pymongo.ASCENDING)
            for doc in docs:
                if doc["sender"] == "human":
                    question_count += 1
                    messages.append(HumanMessage(content=doc["content"]))
                else:
                    messages.append(AIMessage(content=doc["content"]))
            messages = deque(messages, maxlen=settings.CHAT_HISTORY_LIMIT*2) 
            messages = list(messages)
            return messages, question_count
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name, "userId": self.user_id})
            return messages, question_count


    async def get_chat_history(self, filterData, size = None):
        messages = []
        try:
            if size is None:
                size = settings.CHAT_HISTORY_DISPLAY_LIMIT
            pipeline = [
                {"$match": filterData},
                {"$project": ChatProjections.get_all_attribute()},
                {"$sort": {"timestamp": -1}},
                {"$limit": size*2},
                {"$sort": {"timestamp": 1}},
            ]
            docs = self.collection.aggregate(pipeline)
            messages = list(docs)
            return messages
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name, "userId": self.user_id})
            return messages


    async def add_user_message(self, content: str, question_number: int):
        try:
            user_timestamp = get_user_time(timeZone=self.input_timezone)
            self.collection.insert_one({
                "_id": generate_uuid(),
                "user_id": self.user_id, 
                "credit_service_user_id": self.credit_service_user_id, 
                "sender": "human",
                "content": content,
                "sent_by_user": True,
                "question_number": question_number,
                "timestamp": user_timestamp
            })
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name, "userId": self.user_id})
            return False

    async def add_ai_message(self, content: str, question_number: int):
        try:
            user_timestamp = get_user_time(timeZone=self.input_timezone)
            self.collection.insert_one({
                "_id": generate_uuid(),
                "user_id": self.user_id, 
                "credit_service_user_id": self.credit_service_user_id, 
                "sender": "ai",
                "content": content,
                "sent_by_user": False,
                "question_number": question_number,
                "timestamp": user_timestamp
            })
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name, "userId": self.user_id})
            return False

    async def clear(self):
        try:
            self.collection.delete_many({"user_id": self.user_id})
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name, "userId": self.user_id})
            return False

    
    async def delete_chat_history(self, filterData: dict):
        try:
            res = self.collection.delete_many(filterData)
            return res
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name, "userId": self.user_id})
            raise error