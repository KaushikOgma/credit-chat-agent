
import traceback
from app.utils.config import settings
from app.utils.constants import DBCollections
from app.utils.helpers.common_helper import generate_uuid
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.schemas.log_schema import LogSchemasResponse, LogSortFields, SaveLogSchema
from app.attribute_selector.log_attributes import LogProjections
from pymongo.database import Database
from app.utils.logger import setup_logger
logger = setup_logger()


class LogRepository:

    def __init__(self):
        # Get the logger instance
        self.logger = logger
        self.serviceName = "logging_service"


    async def add_log(self, db: Database, data: SaveLogSchema):
        serviceName = None
        userId = None
        try:
            if "id" in data:
                del data["id"]
            newLogId = generate_uuid()
            data["_id"] = newLogId
            data["createdAt"] = get_user_time()
            data["updatedAt"] = get_user_time()
            log_entry = {
                "message": data["message"],
                "type": data.get("type"),
                "stackTrace": data.get("stackTrace"),
                "timestamp": get_user_time()
            }
            log_filter = {}
            if "serviceName" in data and data["serviceName"] is not None:
                serviceName = data["serviceName"]
                log_filter["serviceName"] = data["serviceName"]
            if "userId" in data and data["userId"] is not None:
                userId = data["userId"]
                log_filter["userId"] = data["userId"]
            if len(log_filter) > 0:
                log_filter["moduleName"] = data["moduleName"]

            existing_log = db[DBCollections.LOG.value].find_one(log_filter)
            if len(log_filter) == 0 or not existing_log:
                data["logTrail"] = [log_entry]
                if data["type"].value == "INFO":
                    data["status"] = "SUCCESS"
                else:
                    data["status"] = "ERROR"
                del data["type"]
                del data["stackTrace"]
                db[DBCollections.LOG.value].insert_one(data)
            else:
                existing_log["logTrail"].append(log_entry)
                if data["type"].value == "INFO":
                    existing_log["status"] = "SUCCESS"
                else:
                    existing_log["status"] = "ERROR"
                existing_log["message"] = data["message"]
                existing_log["updatedAt"] = get_user_time()
                del data["type"]
                del data["stackTrace"]
                db[DBCollections.LOG.value].update_one(
                    {"_id": existing_log["_id"]},
                    {"$set": existing_log}
                )
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error

    async def get_log(self, db: Database, filterData: dict, sort_params: dict, input_timezone = None):
        try:
            # Perform the aggregation pipeline
            pipeline = [
                {"$match": filterData},
                # project the fields which need to be included in the final result
                {"$project": LogProjections.get_all_attribute(input_timezone)},
                # sort the fields depending on sort params
                {"$sort": sort_params}
            ]
            data = list(db[DBCollections.LOG.value].aggregate(pipeline))
            return data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.serviceName})
            raise error