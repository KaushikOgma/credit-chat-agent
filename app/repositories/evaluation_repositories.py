
import traceback
from typing import List
from app.attribute_selector.finetune_attributes import FinetuneProjections
from app.utils.config import settings
from app.utils.constants import DBCollections
from app.utils.helpers.common_helper import generate_uuid
from app.utils.helpers.password_helper import hash_password
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.schemas.metadata_schema import MetadataSchema
from app.utils.helpers.auth_helper import generate_api_key
from app.attribute_selector.evaluation_attributes import EvaluationProjections
from pymongo.database import Database
from app.utils.logger import setup_logger
logger = setup_logger()


class EvaluationRepository:

    def __init__(self):
        # Get the logger instance
        self.logger = logger
        self.serviceName = "evaluation"


    async def add_eval_data(self, db: Database, data: List[dict]):
        try:
            inserted_ids = []
            insert_data = []
            for indx, curr_data in enumerate(data):
                is_eval_data_exists = (
                    db[DBCollections.TEST_DATA.value].find_one({
                        "question": curr_data["question"]
                    })
                )
                if is_eval_data_exists:
                    temp_data = curr_data.copy()
                    if "id" in temp_data:
                        del temp_data["id"]
                    if "createdAt" in temp_data:
                        del temp_data["createdAt"]
                    temp_data["updatedAt"] = get_user_time()
                    temp_data["isProcessed"] = False
                    inserted_ids.append(is_eval_data_exists["_id"])
                    db[DBCollections.TEST_DATA.value].update_one({"_id": is_eval_data_exists["_id"]}, {"$set": temp_data})
                else:
                    temp_data = curr_data.copy()
                    if "id" in temp_data:
                        del temp_data["id"]
                    newEvalId = generate_uuid()
                    temp_data["_id"] = newEvalId
                    temp_data["createdAt"] = get_user_time()
                    temp_data["updatedAt"] = get_user_time()
                    temp_data["isProcessed"] = False
                    inserted_ids.append(newEvalId)
                    insert_data.append(temp_data)
            if len(insert_data) > 0:
                db[DBCollections.TEST_DATA.value].insert_many(insert_data)
            return inserted_ids
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error


    async def get_eval_data_details_by_id(self, db: Database, id: str):
        try:
            data = dict(db[DBCollections.TEST_DATA.value].find_one({"_id": id}, EvaluationProjections.get_all_attribute()))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        


    async def update_eval_data(self, db: Database, id: str, data: dict):
        try:
            is_train_ata_exists = (
                db[DBCollections.TEST_DATA.value].find_one({
                    "_id": id
                })
            )
            if is_train_ata_exists:
                if "id" in data:
                    del data["id"]
                data["updatedAt"] = get_user_time()
                db[DBCollections.TEST_DATA.value].update_one({"_id": id}, {"$set": data})
                return True
            else:
                return False
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error


    async def make_eval_data_processed(self, db: Database, eval_ids: list[str]):
        try:
            db[DBCollections.TEST_DATA.value].update_many({"_id": {"$in": eval_ids}}, {"$set": {"isProcessed": True}})
            return True
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error


    async def get_eval_data(self, db: Database, filterData: dict, sort_params: list, input_timezone = None):
        try:
            pipeline = [
                {"$match": filterData},
                {"$project": EvaluationProjections.get_all_attribute(input_timezone)},
                {"$sort": sort_params}
            ]
            data = list(db[DBCollections.TEST_DATA.value].aggregate(pipeline))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        

    async def get_eval_qa_pairs(self, db: Database, qa_pair_ids: List[str]):
        try:
            data = list(db[DBCollections.TEST_DATA.value].find({"_id": {"$in": qa_pair_ids}}, EvaluationProjections.get_qa_attribute()))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        

    async def delete_eval_data(self, db: Database, filterData: dict):
        try:
            res = db[DBCollections.TEST_DATA.value].delete_many(filterData)
            return res
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        

    async def save_eval_result(self, db: Database, model_data_id: str, metrices: dict):
        try:
            is_model_data_exists = (
                db[DBCollections.MODEL_DATA.value].find_one({
                    "_id": model_data_id
                })
            )
            if is_model_data_exists:
                temp_data = {}
                temp_data["metrices"] = metrices
                temp_data["updatedAt"] = get_user_time()
                db[DBCollections.MODEL_DATA.value].update_one({"_id": is_model_data_exists["_id"]}, {"$set": temp_data})
                return False
            else:
                return True
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        



    async def get_model_details_by_id(self, db: Database, id: str):
        try:
            data = dict(db[DBCollections.MODEL_DATA.value].find_one({"_id": id}, FinetuneProjections.get_model_attribute()))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        

