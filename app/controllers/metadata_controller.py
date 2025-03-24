from fastapi.responses import JSONResponse
from tqdm import tqdm
from app.controllers.evaluation_controller import EvaluationController
from app.controllers.finetune_controller import FinetuneController
from app.repositories.evaluation_repositories import EvaluationRepository
from app.repositories.finetune_repositories import FinetuneRepository
from app.schemas.evaluation_schema import EvalQASchema
from app.schemas.finetune_schema import TrainQASchema
from app.services.data_ingestor import DataIngestor
from app.services.qa_evaluator import QAEvaluator
from app.services.qa_generator import QAGenerator
from app.utils.helpers.date_helper import get_user_time, convert_timezone
from app.repositories.metadata_repositories import MetadataRepository
from datetime import datetime
from app.utils.config import settings
from fastapi.exceptions import HTTPException
from app.utils.logger import setup_logger
logger = setup_logger()


class MetadataController:

    def __init__(self, metadata_repo: MetadataRepository, finetune_repo: FinetuneRepository,eval_repo: EvaluationRepository, finetune_controller: FinetuneController, eval_controller: EvaluationController, data_ingestor: DataIngestor, qa_generator: QAGenerator, qa_evaluator:QAEvaluator):
        self.metadata_repo = metadata_repo
        self.finetune_repo = finetune_repo
        self.eval_repo = eval_repo
        self.data_ingestor = data_ingestor
        self.qa_generator = qa_generator
        self.qa_evaluator = qa_evaluator
        self.finetune_controller = finetune_controller
        self.eval_controller = eval_controller
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
            print("sort_params:: ",sort_params)
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
            return data
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
            return data
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
            # add the data into metadata collection
            inserted_id = await self.metadata_repo.add_metadata(db, data)
            # generate qa pairs for the content
            qa_pairs = await self.qa_generator.generate_question_and_answer(data["content"])
            if data["isTrainData"]:
                train_data = []
                # check if the data is train data then insert qa pairs into train data collection
                for pair in tqdm(qa_pairs, desc="Processing Questions"):
                    train_data_entry = TrainQASchema(
                        question = pair["question"],
                        answer = pair["answer"],
                        metadataId = str(inserted_id),
                        isActive = True
                    )
                    train_data.append(train_data_entry.model_dump())
                await self.finetune_controller.add_train_data(train_data, db)
            else:
                eval_data = []
                # check if the data is eval data then insert qa pairs into eval data collection
                for pair in tqdm(qa_pairs, desc="Processing Questions"):
                    eval_data_entry = EvalQASchema(
                        question = pair["question"],
                        answer = pair["answer"],
                        metadataId = str(inserted_id),
                        isProcessed = False,
                        isActive = True
                    )
                    eval_data.append(eval_data_entry.model_dump())
                await self.eval_controller.add_eval_data(eval_data, db)
            # update the meta as processed
            await self.metadata_repo.make_metadata_processed(db, [inserted_id])
            return inserted_id
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
            metadata = await self.metadata_repo.get_metadata_details_by_id(db, id)
            isTrainData = data["isTrainData"] if "isTrainData" in data else metadata["isTrainData"]
            if "content" in data and data["content"] is not None:
                qa_pairs = await self.qa_generator.generate_question_and_answer(data["content"])
                if isTrainData:
                    # delete existing qa pairse as the content is updated
                    await self.finetune_repo.delete_tarin_data(db, {"metadataId": str(id)})
                    train_data = []
                    # check if the data is train data then insert qa pairs into train data collection
                    for pair in tqdm(qa_pairs, desc="Processing Questions"):
                        train_data_entry = TrainQASchema(
                            question = pair["question"],
                            answer = pair["answer"],
                            metadataId = str(id),
                            isActive = True
                        )
                        train_data.append(train_data_entry.model_dump())
                    await self.finetune_controller.add_train_data(train_data, db)
                else:
                    # delete existing qa pairse as the content is updated
                    await self.eval_repo.delete_eval_data(db, {"metadataId": str(id)})
                    eval_data = []
                    # check if the data is eval data then insert qa pairs into eval data collection
                    for pair in tqdm(qa_pairs, desc="Processing Questions"):
                        eval_data_entry = EvalQASchema(
                            question = pair["question"],
                            answer = pair["answer"],
                            metadataId = str(id),
                            isProcessed = False,
                            isActive = True
                        )
                        eval_data.append(eval_data_entry.model_dump())
                    await self.eval_controller.add_eval_data(eval_data, db)
            update_flag = await self.metadata_repo.update_Metadata(db, id, data)
            # update the meta as processed
            await self.metadata_repo.make_metadata_processed(db, [id])
            return update_flag
        except Exception as error:
            logger.exception(error)
            raise error
        

    async def delete_metadatas(
        self,
        db,
        startDate: datetime,
        endDate: datetime,
        fileName: str,
        isTrainData: bool,
        isProcessed: bool,
        deleteQAPaires: bool
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
                filterData["createdAt"] = {
                    '$gte': convert_timezone(startDate, to_string=False, timeZone="UTC"),
                }
                if endDate is None:
                    endDate = startDate.replace(hour=23, minute=59, second=59)
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
                else:
                    filterData["createdAt"]["$lte"] = convert_timezone(endDate, to_string=False, timeZone="UTC")
            if deleteQAPaires:
                sort_order = {
                    "createdAt": -1
                }
                data = await self.metadata_repo.get_metadatas(db, filterData, sort_order, input_timezone)
                # Delete associated qa data related to the metadata
                for elm in data:
                    if elm["isTrainData"]:
                        await self.finetune_controller.delete_train_data(
                            db,
                            startDate = None, 
                            endDate = None, 
                            fileName = elm["fileName"] if "fileName" in elm else None, 
                            isActive = None
                        )
                    else:
                        await self.eval_controller.delete_eval_data(
                            db,
                            startDate = None, 
                            endDate = None, 
                            fileName = elm["fileName"] if "fileName" in elm else None, 
                            isProcessed = None,
                            isActive = None
                        )
                    filterMetadata = {
                        "fileName": elm["fileName"] if "fileName" in elm else None
                    }
                    await self.metadata_repo.delete_metadata(db, filterMetadata)
            else:
                data = await self.metadata_repo.delete_metadata(db, filterData)
            return True
        except Exception as error:
            logger.exception(error)
            raise error
    