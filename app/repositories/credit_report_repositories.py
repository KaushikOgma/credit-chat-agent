
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
from app.attribute_selector.credit_report_attributes import CreditReportProjections
from pymongo.database import Database
from app.utils.logger import setup_logger
logger = setup_logger()


class CreditReportRepository:

    def __init__(self):
        # Get the logger instance
        self.logger = logger
        self.serviceName = "credit_report_repository"



    async def add_report(self, db: Database, data: dict):
        try:
            if "userId" in data and data["userId"] is not None:
                is_report_exists = (
                    db[DBCollections.CREDIT_REPORT.value].find_one({
                        "userId": data["userId"]
                    })
                )
            else:
                is_report_exists = None
            if is_report_exists:
                temp_data = data.copy()
                if "id" in temp_data:
                    del temp_data["id"]
                if "createdAt" in temp_data:
                    del temp_data["createdAt"]
                temp_data["updatedAt"] = get_user_time()
                db[DBCollections.CREDIT_REPORT.value].update_one({"_id": is_report_exists["_id"]}, {"$set": temp_data})
                inserted_id = is_report_exists["_id"]
            else:
                if "id" in data:
                    del data["id"]
                newReportId = generate_uuid()
                data["_id"] = newReportId
                data["createdAt"] = get_user_time()
                data["updatedAt"] = get_user_time()
                result = db[DBCollections.CREDIT_REPORT.value].insert_one(data)
                # Get the inserted id to return to call copy config api from the ui after user creation
                inserted_id = result.inserted_id
            return inserted_id
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error

    async def update_report(self, db: Database, id: str, data: dict):
        try:
            is_report_exists = (
                db[DBCollections.CREDIT_REPORT.value].find_one({
                    "_id": id
                })
            )
            if is_report_exists:
                if "id" in data:
                    del data["id"]
                data["updatedAt"] = get_user_time()
                db[DBCollections.CREDIT_REPORT.value].update_one({"_id": id}, {"$set": data})
                return True
            else:
                return False
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        

    async def get_reports(self, db: Database, filterData: dict, sort_params: list, input_timezone = None):
        try:
            pipeline = [
                {"$match": filterData},
                {"$project": CreditReportProjections.get_silected_attribute(input_timezone)},
                {"$sort": sort_params}
            ]
            data = list(db[DBCollections.CREDIT_REPORT.value].aggregate(pipeline))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error
        