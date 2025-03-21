from fastapi.responses import JSONResponse
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.repositories.metadata_repositories import MetadataRepository
from datetime import datetime
from app.utils.config import settings
from fastapi.exceptions import HTTPException
from app.utils.logger import setup_logger
logger = setup_logger()


class MetadataController:

    def __init__(self, metadata_repo: MetadataRepository):
        self.metadata_repo = metadata_repo
        self.service_name = "metadata_manage_service"

    async def get_metadatas(
        self,
        db,
        startDate: datetime,
        endDate: datetime,
        fileName: str,
        isTrainData: bool,
        isProcessed: bool,
        sort_params: list,
    ) -> dict:
        try:
            filterData = {}
            input_timezone = None
            if fileName is not None:
                filterData["fileName"] = fileName
            if isTrainData is not None:
                filterData["isTrainData"] = isTrainData
            if isProcessed is not None:
                filterData["isProcessed"] = isProcessed
            if startDate is not None:
                input_timezone = startDate.tzname().replace("UTC","")
                filterData["createdAt"] = {
                    '$gte': convert_timezone(startDate, to_string=False, timeZone="UTC"),
                }
                if endDate is None:
                    endDate = startDate.replace(hour=23, minute=59, second=59)
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
                else:
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
            data = await self.metadata_repo.get_metadatas(db, filterData, sort_params, input_timezone)
            return JSONResponse(
                        status_code=200, content={"data": data, "message": "Data fetched successfully"}
                    )
        except Exception as error:
            logger.exception(error)
            raise error
    

    async def get_metadata_detail(self, db, id):
        """**Summary:**
        This method is responsible for fetching all users.

        **Args:**
        - `db` (Database): db session referance.
        """
        try:
            data = await self.metadata_repo.get_metadata_details_by_id(db, id)
            return JSONResponse(
                status_code=200, content={"data": data, "message": "Data fetched successfully"}
            )
        except Exception as error:
            logger.exception(error)
            raise error


    async def add_metadata(self, data, db):
        """**Summary:**
        This method is responsible for adding a user with an API key.

        **Args:**
        - `data` (Dict): metadata to be inserted.
        - `db` (Database): db session referance.
        """
        try:
            inserted_id = await self.metadata_repo.add_metadata(db, data)
            return JSONResponse(
                        status_code=200, content={"id":inserted_id, "message": "Data inserted successfully"}
                    )
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error


    async def update_metadata(self, id, data, db):
        """**summary**
        A method to update an existing user based on the provided user id and user data.

        **Args:**
        - `id` (int): The user id to update.
        - `data` (Dict): The metadata to update.
        - `db` (Database): The database session reference.

        **Returns:**
        - A message dict indicating the success of the update operation.
        - If the user id is invalid, returns a JSONResponse with a 401 status code and a message indicating an invalid user id.
        """
        try:
            # print("id:: ",id)
            # print("user_data:: ",user_data)
            # Check if order exists or not
            update_flag = await self.metadata_repo.update_Metadata(db, id, data)

            if update_flag:
                return JSONResponse(
                    status_code=200, content={"message": "Data updated successfully"}
                )
            else:
                return JSONResponse(
                    status_code=400, content={"message": "Invalid metadata id"}
                )
        except Exception as error:
            logger.exception(error)
            raise error