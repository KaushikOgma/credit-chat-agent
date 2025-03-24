import asyncio
from fastapi import Depends
from fastapi.responses import JSONResponse
from app.db import get_db
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.repositories.finetune_repositories import FinetuneRepository
from app.repositories.evaluation_repositories import EvaluationRepository
from app.services.qa_evaluator import QAEvaluator
from datetime import datetime
from pymongo.database import Database
from app.utils.config import settings
from fastapi.exceptions import HTTPException
from app.utils.logger import setup_logger
logger = setup_logger()


class EvaluationController:

    def __init__(self, eval_repo: EvaluationRepository, question_evaluator: QAEvaluator):
        self.eval_repo = eval_repo
        self.question_evaluator = question_evaluator
        self.service_name = "evaluation"

    async def get_eval_data(
        self,
        db,
        startDate: datetime,
        endDate: datetime,
        fileName: str,
        isActive: bool,
        isProcessed: bool,
        sort_params: list,
    ) -> dict:
        try:
            filterData = {}
            input_timezone = None
            if fileName is not None:
                filterData["fileName"] = fileName
            if isActive is not None:
                filterData["isActive"] = isActive
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
            data = await self.eval_repo.get_eval_data(db, filterData, sort_params, input_timezone)
            return data
        except Exception as error:
            logger.exception(error)
            raise error
    
    async def add_eval_data(self, data, db):
        """**Summary:**
        This method is responsible for adding a user with an API key.

        **Args:**
        - `data` (Dict): metadata to be inserted.
        - `db` (Database): db session referance.
        """
        try:
            inserted_ids = await self.eval_repo.add_eval_data(db, data)
            qa_pairs = await self.eval_repo.get_eval_qa_pairs(db, inserted_ids)
            isProcessed = await self.question_evaluator.sync_vector_db(qa_pairs=qa_pairs)
            if isProcessed:
                await self.eval_repo.make_eval_data_processed(db, inserted_ids)
            return inserted_ids
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error


    async def update_eval_data(self, id, data, db):
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
            update_flag = await self.eval_repo.update_eval_data(db, id, data)
            return update_flag
        except Exception as error:
            logger.exception(error)
            raise error
        

    async def delete_eval_data(
        self,
        db,
        startDate: datetime,
        endDate: datetime,
        fileName: str,
        isProcessed: bool,
        isActive: bool,
    ) -> dict:
        try:
            filterData = {}
            if fileName is not None:
                filterData["fileName"] = fileName
            if isActive is not None:
                filterData["isActive"] = isActive
            if isProcessed is not None:
                filterData["isProcessed"] = isProcessed
            if startDate is not None:
                filterData["createdAt"] = {
                    '$gte': convert_timezone(startDate, to_string=False, timeZone="UTC"),
                }
                if endDate is None:
                    endDate = startDate.replace(hour=23, minute=59, second=59)
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
                else:
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
            data = await self.eval_repo.get_eval_data(db, filterData, {"createdAt": -1})
            ids_list = [elm["_id"] for elm in data]
            print("delete_eval_data_from_vector_db:: ",ids_list)
            has_deleted = await self.question_evaluator.delete_eval_data_from_vector_db(id_list=ids_list)
            if has_deleted:
                data = await self.eval_repo.delete_eval_data(db, filterData)
            return True
        except Exception as error:
            logger.exception(error)
            raise error
        

    async def initiate_evaluating(
        self,
        db,
        startDate: datetime,
        endDate: datetime,
        fileName: str,
        isActive: bool,
        model_id: str,
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
                self.start_evaluating(filterData, model_id)
            )
            return True
        except Exception as error:
            logger.exception(error)
            raise error
        

        
    async def start_evaluating(
            self, 
            filter_data, 
            model_id,
            db_instance: Database = Depends(get_db)
        ):
        try:       
            async with db_instance as db:
                qa_data = await self.eval_repo.get_eval_data(db, filter_data, {"createdAt": -1})
                ids_list = [elm["_id"] for elm in qa_data]
                qa_data = await self.eval_repo.get_eval_qa_pairs(db, ids_list)
                result = await self.question_evaluator.evaluate_qa_pairs(qa_data)
                if result:
                    await self.eval_repo.save_eval_result(db, model_id, result["agg_scores"])
        except Exception as error:
            logger.exception(error)
            raise error