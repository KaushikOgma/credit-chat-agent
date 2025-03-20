"""
Setting up the logger so that it can add log to the db
usage examples:
logger.info("message", extra={"moduleName": "credit_genius_ai", "serviceName": "qa_generator"})
logger.error(error, extra={"moduleName": "credit_genius_ai", "serviceName": "qa_generator"})
logger.exception(error, extra={"moduleName": "credit_genius_ai", "serviceName": "qa_generator"})

"""
import logging
import sys
from app.schemas.log_schema import SaveLogSchema
from app.db import get_db_instance
from app.utils.config import settings
from app.utils.constants import DBCollections
from app.utils.helpers.date_helper import get_user_time
from app.utils.helpers.common_helper import generate_uuid

class DBLogHandler(logging.Handler):
    def emit(self, record):
        """
        **Summary:**
        Emit a log entry to the database.
        This function takes a log record and emits it to the database. It creates a log entry dictionary based on the record's attributes, including the message, type, stack trace, module name, request ID, and order ID. It then retrieves the module ID from the database based on the module name. The log entry dictionary is updated with the module ID, creation timestamp, and update timestamp.
        If there are no filters specified for the log entry (no request ID or order ID), the log entry is inserted into the database as a new document. If there are filters specified, the function checks if a log entry with the same filters already exists in the database. If not, a new log entry is inserted with the appropriate log type (INFO or ERROR) and the log entry is added to the corresponding log list. If a log entry with the same filters already exists, the log entry is appended to the existing log entry's log list and the update timestamp is updated.
        If there is an error while logging to the database, an exception is raised.

        **Args:**
        - `record` (LogRecord): The log record to emit.

        **Raises:**
        - `Exception`: If there is an error while logging to the database.

        **Note:**
        - The function closes the database connection after logging.
        """
        db = get_db_instance()
        try:
            log_entry = SaveLogSchema(
                message = record.getMessage(),
                type = record.levelname.upper(),
                stackTrace = self.format(record) if record.exc_info else None,
                moduleName = getattr(record, 'moduleName', str(settings.MODULE)),
                serviceName = getattr(record, 'serviceName', None),
                userId = getattr(record, 'userId', None)
            )
            log_entry_data = log_entry.model_dump()
            if "id" in log_entry_data:
                del log_entry_data["id"]
            newLogId = generate_uuid()
            log_entry_data["_id"] = newLogId
            log_entry_data["createdAt"] = get_user_time()
            log_entry_data["updatedAt"] = get_user_time()
            log_entry = {
                "message": log_entry_data["message"],
                "type": log_entry_data["type"],
                "stackTrace": log_entry_data["stackTrace"],
                "timestamp": get_user_time()
            }
            log_filter = {}
            if "serviceName" in log_entry_data and log_entry_data["serviceName"] is not None:
                log_entry_data["serviceName"] = str(log_entry_data["serviceName"])
                log_filter["serviceName"] = log_entry_data["serviceName"]
            if "userId" in log_entry_data and log_entry_data["userId"] is not None:
                log_entry_data["userId"] = int(log_entry_data["userId"])
                log_filter["userId"] = log_entry_data["userId"]
            if len(log_filter) > 0:
                log_filter["moduleName"] = log_entry_data["moduleName"]

            existing_log = db[DBCollections.LOG.value].find_one(log_filter)
            if len(log_filter) == 0 or not existing_log:
                log_entry_data["logTrail"] = [log_entry]
                if log_entry_data["type"] == "INFO":
                    log_entry_data["status"] = "SUCCESS"
                else:
                    log_entry_data["status"] = "ERROR"
                del log_entry_data["type"]
                del log_entry_data["stackTrace"]
                db[DBCollections.LOG.value].insert_one(log_entry_data)
            else:
                existing_log["logTrail"].append(log_entry)
                if log_entry_data["type"] == "INFO":
                    existing_log["status"] = "SUCCESS"
                else:
                    existing_log["status"] = "ERROR"
                existing_log["message"] = log_entry_data["message"]
                existing_log["updatedAt"] = get_user_time()
                del log_entry_data["type"]
                del log_entry_data["stackTrace"]
                db[DBCollections.LOG.value].update_one(
                    {"_id": existing_log["_id"]},
                    {"$set": existing_log}
                )
        except Exception as e:
            # print(f"Failed to log to database: {e}")
            raise e
        finally:
            # Close the db connection
            db.client.close()


def setup_logger():
    """
    **Summary:**
    Set up the logger for the module.
    This function initializes the logger for the module specified in the `settings.MODULE` variable. If the logger does not have any handlers, it sets the logger's level to `logging.INFO` and creates two handlers: a `DBLogHandler` for logging to the database and a `StreamHandler` for logging to the console. The handlers are added to the logger.

    **Returns:**
    - `logger` (logging.Logger): The logger object for the module.

    **Raises:**
    - `Exception`: If an error occurs while setting up the logger.

    """
    try:
        logger = logging.getLogger(settings.MODULE)
        
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # Create DBLogHandler for logging to the database
            db_handler = DBLogHandler()
            db_handler.setFormatter(formatter)
            logger.addHandler(db_handler)

            
            # Create StreamHandler for logging to the console
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger
    except Exception as e:
        # print(f"setup_logger:: Failed to setup the logger: {e}")
        raise e