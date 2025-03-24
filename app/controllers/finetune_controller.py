import asyncio
from fastapi import Depends
from fastapi.responses import JSONResponse
from app.db import get_db
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.services.llm_finetune import OpenAIFineTuner
from app.repositories.finetune_repositories import FinetuneRepository
from datetime import datetime
from pymongo.database import Database
from app.utils.config import settings
from fastapi.exceptions import HTTPException
from app.utils.logger import setup_logger
logger = setup_logger()


class FinetuneController:

    def __init__(self, finetune_repo: FinetuneRepository, opeai_finetuner: OpenAIFineTuner):
        self.finetune_repo = finetune_repo
        self.opeai_finetuner = opeai_finetuner
        self.service_name = "finetune"

    async def get_train_data(
        self,
        db,
        startDate: datetime,
        endDate: datetime,
        fileName: str,
        isActive: bool,
        sort_params: list,
    ) -> dict:
        try:
            filterData = {}
            input_timezone = None
            if fileName is not None:
                filterData["fileName"] = fileName
            if isActive is not None:
                filterData["isActive"] = isActive
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
            data = await self.finetune_repo.get_tarin_data(db, filterData, sort_params, input_timezone)
            return data
        except Exception as error:
            logger.exception(error)
            raise error
    
    async def add_train_data(self, data, db):
        """**Summary:**
        This method is responsible for adding a user with an API key.

        **Args:**
        - `data` (Dict): metadata to be inserted.
        - `db` (Database): db session referance.
        """
        try:
            inserted_ids = await self.finetune_repo.add_train_data(db, data)
            return inserted_ids
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error


    async def update_train_data(self, id, data, db):
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
            update_flag = await self.finetune_repo.update_tarin_data(db, id, data)
            return update_flag
        except Exception as error:
            logger.exception(error)
            raise error
        

    async def delete_train_data(
        self,
        db,
        startDate: datetime,
        endDate: datetime,
        fileName: str,
        isActive: bool
    ) -> dict:
        try:
            filterData = {}
            if fileName is not None:
                filterData["fileName"] = fileName
            if isActive is not None:
                filterData["isActive"] = isActive
            if startDate is not None:
                filterData["createdAt"] = {
                    '$gte': convert_timezone(startDate, to_string=False, timeZone="UTC"),
                }
                if endDate is None:
                    endDate = startDate.replace(hour=23, minute=59, second=59)
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
                else:
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
            data = await self.finetune_repo.delete_tarin_data(db, filterData)
            return True
        except Exception as error:
            logger.exception(error)
            raise error
    


    async def initiate_training(
        self,
        db,
        startDate: datetime,
        endDate: datetime,
        fileName: str,
        isActive: bool
    ) -> dict:
        try:
            filterData = {}
            if fileName is not None:
                filterData["fileName"] = fileName
            if isActive is not None:
                filterData["isActive"] = isActive
            if startDate is not None:
                filterData["createdAt"] = {
                    '$gte': convert_timezone(startDate, to_string=False, timeZone="UTC"),
                }
                if endDate is None:
                    endDate = startDate.replace(hour=23, minute=59, second=59)
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
                else:
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
            asyncio.create_task(
                self.start_training(filterData)
            )
            return True
        except Exception as error:
            logger.exception(error)
            raise error
        

        
    async def start_training(
            self, 
            filter_data, 
            db_instance: Database = Depends(get_db)
        ):
        try:       
            async with db_instance as db:
                qa_data = await self.finetune_repo.get_tarin_data(db, filter_data, {"createdAt": -1})
                ids_list = [elm["_id"] for elm in qa_data]
                qa_data = await self.finetune_repo.get_train_qa_pairs(db, ids_list)
                model_id = await self.opeai_finetuner.start_finetune(qa_data)
                if model_id:
                    await self.finetune_repo.save_model(db, model_id)
        except Exception as error:
            logger.exception(error)
            raise error