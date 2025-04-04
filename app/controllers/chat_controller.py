from fastapi.responses import JSONResponse
from app.repositories.model_data_repositories import ModelDataRepository
from app.schemas.chat_schema import ChatAgentRequest, ChatRequest
from app.services.chat_service import ChatService
from app.services.chat_agent_service import runnable_graph
from tqdm import tqdm
from pymongo.database import Database
from app.utils.config import settings
from app.utils.logger import setup_logger

logger = setup_logger()

class ChatController:
    
    def __init__(self, chat_service: ChatService, model_data_repo: ModelDataRepository):
        self.chat_service = chat_service
        self.model_data_repo = model_data_repo
        self.service_name = "chat_service"

    async def test_chat(self, db: Database, req_data: ChatRequest, model_data_id: str):
        try:
            resp = {
                "question": req_data.question,
                "answer": None
            }
            model_data = await self.model_data_repo.get_model_details_by_id(db, model_data_id)
            if model_data:
                response = await self.chat_service.get_response(question=req_data.question, model_id=model_data["model_id"])
                resp["answer"] = response
            return resp
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error


    async def test_chat_agent(self, req_data: ChatAgentRequest):
        try:
            input = req_data.model_dump()
            print(f"[DEBUG] input: {input}")  # Debug
            result = await runnable_graph.ainvoke(input)
            print("[DEBUG] Final Output:", result.get("answer")) 
            print("Travarsed Path:: ", " --> ".join(elm for elm in result.get("path",[])))
            return {"question_number": result.get("question_number", None),"question": result.get("question", input["user_query"]),"answer": result.get("answer"), "traversed_path": " --> ".join(elm for elm in result.get("path",[]))}
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            raise error
