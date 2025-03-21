
import traceback
from app.utils.config import settings
from app.utils.constants import DBCollections
from app.utils.helpers.common_helper import generate_uuid
from app.utils.helpers.password_helper import hash_password
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.schemas.user_schema import UserSchemasResponse, UserSortFields, SaveUserSchema
from app.utils.helpers.auth_helper import generate_api_key
from app.attribute_selector.user_attributes import UserProjections
from pymongo.database import Database
from app.utils.logger import setup_logger
logger = setup_logger()


class UserRepository:

    def __init__(self):
        # Get the logger instance
        self.logger = logger
        self.serviceName = "user_manage_service"


    async def add_user(self, db: Database, data: dict):
        try:
            if "id" in data:
                del data["id"]
            data["password"] = await hash_password(data["password"])
            newUserId = generate_uuid()
            data["_id"] = newUserId
            data["createdAt"] = get_user_time()
            data["updatedAt"] = get_user_time()

            # Generate and add API key
            api_key = generate_api_key()
            data["apiKey"] = api_key

            result = db[DBCollections.USER.value].insert_one(data)
            # Get the inserted id to return to call copy config api from the ui after user creation
            inserted_id = result.inserted_id
            return inserted_id
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error

    async def update_user(self, db: Database, id: str, data: dict):
        try:
            is_user_exists = (
                db[DBCollections.USER.value].find_one({
                    "_id": id
                })
            )
            if is_user_exists:
                if "id" in data:
                    del data["id"]
                data["updatedAt"] = get_user_time()
                db[DBCollections.USER.value].update_one({"_id": id}, {"$set": data})
                return True
            else:
                return False
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error



    async def get_user_details_by_id(self, db: Database, userId: str):
        try:
            data = dict(db[DBCollections.USER.value].find_one({"_id": userId}, UserProjections.get_all_attribute()))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        


    async def get_users(self, db: Database, filterData: dict, sort_params: list, input_timezone = None):
        try:
            pipeline = [
                {"$match": filterData},
                {"$project": {
                    **UserProjections.get_silected_attribute(timezone=input_timezone),
                    "config": 1,
                    "updatedAt": {
                        "$ifNull": ["$updatedAt", None]
                    }
                }},
                {"$sort": sort_params}
            ]
            data = list(db[DBCollections.USER.value].aggregate(pipeline))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error