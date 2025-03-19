from app.services.qa_generator import QAGenerator
from app.schemas.module_schema import QAGenerateRequest
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()



async def generate_qa(req_data: QAGenerateRequest):
    serviceName = "qa_generator"
    try:
        text = req_data.text
        qa_generator = QAGenerator()
        qa = await qa_generator.generate_question_and_answer(text)
        return qa
    except Exception as error:
        logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": serviceName})
        return {"error": str(error)}