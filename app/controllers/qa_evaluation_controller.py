from app.services.qa_evaluator import QAEvaluator
from app.schemas.module_schema import QAEvaluationRequest
from app.utils.config import settings
from app.utils.logger import setup_logger
logger = setup_logger()


async def get_evaluation_score(req_data: QAEvaluationRequest):
    serviceName = "qa_evaluator"
    try:
        qa_pairs = [item.model_dump() for item in req_data]
        qa_evaluator = QAEvaluator()
        score = await qa_evaluator.evaluate_qa_pairs(qa_pairs)
        return score
    except Exception as error:
        logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": serviceName})
        return {"error": str(error)}