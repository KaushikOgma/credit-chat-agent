
import traceback
from app.utils.config import settings
from app.utils.constants import DBCollections
from app.utils.helpers.common_helper import generate_uuid
from app.utils.helpers.password_helper import hash_password
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.schemas.metadata_schema import MetadataSchema
from app.utils.helpers.auth_helper import generate_api_key
from app.attribute_selector.metadata_attributes import MetadataProjections
from pymongo.database import Database
from app.utils.logger import setup_logger
logger = setup_logger()


class MetadataRepository:

    def __init__(self):
        # Get the logger instance
        self.logger = logger
        self.serviceName = "metadata_manage_service"


    async def add_metadata(self, db: Database, data: dict):
        try:
            if "id" in data:
                del data["id"]
            newMetadataId = generate_uuid()
            data["_id"] = newMetadataId
            data["createdAt"] = get_user_time()
            data["updatedAt"] = get_user_time()
            result = db[DBCollections.METADATA.value].insert_one(data)
            # Get the inserted id to return to call copy config api from the ui after user creation
            inserted_id = result.inserted_id
            return inserted_id
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error

    async def update_Metadata(self, db: Database, id: str, data: dict):
        try:
            is_metadata_exists = (
                db[DBCollections.METADATA.value].find_one({
                    "_id": id
                })
            )
            if is_metadata_exists:
                if "id" in data:
                    del data["id"]
                data["updatedAt"] = get_user_time()
                db[DBCollections.METADATA.value].update_one({"_id": id}, {"$set": data})
                return True
            else:
                return False
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error



    async def get_metadata_details_by_id(self, db: Database, userId: str):
        try:
            data = dict(db[DBCollections.METADATA.value].find_one({"_id": userId}, MetadataProjections.get_all_attribute()))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        


    async def get_metadatas(self, db: Database, filterData: dict, sort_params: list, input_timezone = None):
        try:
            pipeline = [
                {"$match": filterData},
                {"$project": MetadataProjections.get_silected_attribute(input_timezone)},
                {"$sort": sort_params}
            ]
            data = list(db[DBCollections.METADATA.value].aggregate(pipeline))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error