
import traceback
from typing import List, Union
from app.utils.config import settings
from app.utils.constants import DBCollections
from app.utils.helpers.common_helper import generate_uuid
from app.utils.helpers.password_helper import hash_password
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.schemas.metadata_schema import MetadataSchema
from app.utils.helpers.auth_helper import generate_api_key
from app.schemas.finetune_schema import ModelDataSchema
from app.attribute_selector.finetune_attributes import FinetuneProjections
from pymongo.database import Database
from app.utils.logger import setup_logger
logger = setup_logger()


class FinetuneRepository:

    def __init__(self):
        # Get the logger instance
        self.logger = logger
        self.serviceName = "finetune"


    async def add_train_data(self, db: Database, data: List[dict]):
        try:
            inserted_ids = []
            insert_data = []
            for curr_data in data:
                is_train_data_exists = (
                    db[DBCollections.TRAIN_DATA.value].find_one({
                        "question": curr_data["question"]
                    })
                )
                if is_train_data_exists:
                    temp_data = curr_data.copy()
                    if "id" in temp_data:
                        del temp_data["id"]
                    if "createdAt" in temp_data:
                        del temp_data["createdAt"]
                    temp_data["updatedAt"] = get_user_time()
                    inserted_ids.append(is_train_data_exists["_id"])
                    temp_data["isProcessed"] = False
                    db[DBCollections.TRAIN_DATA.value].update_one({"_id": is_train_data_exists["_id"]}, {"$set": temp_data})
                else:
                    temp_data = curr_data.copy()
                    if "id" in temp_data:
                        del temp_data["id"]
                    newTraindataId = generate_uuid()
                    temp_data["_id"] = newTraindataId
                    temp_data["createdAt"] = get_user_time()
                    temp_data["updatedAt"] = get_user_time()
                    temp_data["isProcessed"] = False
                    inserted_ids.append(newTraindataId)
                    insert_data.append(temp_data)
            if len(insert_data) > 0:
                db[DBCollections.TRAIN_DATA.value].insert_many(insert_data)
            return inserted_ids
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error


    async def get_fintune_data_details_by_id(self, db: Database, id: str):
        try:
            data = dict(db[DBCollections.TRAIN_DATA.value].find_one({"_id": id}, FinetuneProjections.get_all_attribute()))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        

    async def update_tarin_data(self, db: Database, id: str, data: dict):
        try:
            is_train_ata_exists = (
                db[DBCollections.TRAIN_DATA.value].find_one({
                    "_id": id
                })
            )
            if is_train_ata_exists:
                if "id" in data:
                    del data["id"]
                data["updatedAt"] = get_user_time()
                db[DBCollections.TRAIN_DATA.value].update_one({"_id": id}, {"$set": data})
                return True
            else:
                return False
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error

    async def make_train_data_processed(self, db: Database, train_ids: list[str]):
        try:
            db[DBCollections.TRAIN_DATA.value].update_many({"_id": {"$in": train_ids}}, {"$set": {"isProcessed": True}})
            return True
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error


    async def get_tarin_data(self, db: Database, filterData: dict, sort_params: list, input_timezone = None):
        try:
            pipeline = [
                {"$match": filterData},
                {"$project": FinetuneProjections.get_all_attribute(input_timezone)},
                {"$sort": sort_params}
            ]
            data = list(db[DBCollections.TRAIN_DATA.value].aggregate(pipeline))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        


    async def get_train_qa_pairs(self, db: Database, qa_pair_ids: List[str]):
        try:
            data = list(db[DBCollections.TRAIN_DATA.value].find({"_id": {"$in": qa_pair_ids}}, FinetuneProjections.get_qa_attribute()))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        

    async def delete_tarin_data(self, db: Database, filterData: dict):
        try:
            res = db[DBCollections.TRAIN_DATA.value].delete_many(filterData)
            return res
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        


    async def get_models(self, db: Database):
        try:
            data = list(db[DBCollections.MODEL_DATA.value].find({}, FinetuneProjections.get_model_attribute()).sort("createdAt", -1))                       
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        


    async def save_model(
            self, 
            db: Database, 
            dataset_ids: list[str], 
            file_id: str,
            job_id: str,
            model_id: str,
            params: Union[dict, None],
        ):
        try:
            is_model_data_exists = (
                db[DBCollections.MODEL_DATA.value].find_one({
                    "model_id": model_id
                })
            )
            if is_model_data_exists:
                temp_data = {}
                temp_data["datasetIds"] = dataset_ids
                temp_data["file_id"] = file_id
                temp_data["job_id"] = job_id
                temp_data["params"] = params
                temp_data["updatedAt"] = get_user_time()
                db[DBCollections.MODEL_DATA.value].update_one({"_id": is_model_data_exists["_id"]}, {"$set": temp_data})
                inserted_id = is_model_data_exists["_id"]
            else:
                temp_data = {}
                newModelId = generate_uuid()
                temp_data["_id"] = newModelId
                temp_data["datasetIds"] = dataset_ids
                temp_data["file_id"] = file_id
                temp_data["job_id"] = job_id
                temp_data["model_id"] = model_id
                temp_data["params"] = params
                temp_data["createdAt"] = get_user_time()
                temp_data["updatedAt"] = get_user_time()
                inserted_id = db[DBCollections.MODEL_DATA.value].insert_one(temp_data)
            return inserted_id
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        

