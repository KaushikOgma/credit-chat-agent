
import traceback
from typing import List, Union
from app.utils.config import settings
from app.utils.constants import DBCollections
from app.utils.helpers.common_helper import generate_uuid
from app.utils.helpers.password_helper import hash_password
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.schemas.metadata_schema import MetadataSchema
from app.utils.helpers.auth_helper import generate_api_key
from app.schemas.model_data_schema import ModelDataSchema
from app.attribute_selector.model_data_attributes import ModelDataProjections
from pymongo.database import Database
from app.utils.logger import setup_logger
logger = setup_logger()


class ModelDataRepository:

    def __init__(self):
        # Get the logger instance
        self.logger = logger
        self.serviceName = "model_data_repository"

    async def get_models(self, db: Database):
        try:
            data = list(db[DBCollections.MODEL_DATA.value].find({}, ModelDataProjections.get_all_attribute()).sort("createdAt", -1))                       
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
            data = dict(db[DBCollections.MODEL_DATA.value].find_one({"_id": id}, ModelDataProjections.get_all_attribute()))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error