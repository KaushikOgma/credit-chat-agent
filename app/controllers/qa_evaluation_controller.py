import io
import os
import tempfile
from app.services.qa_evaluator import QAEvaluator
from app.schemas.module_schema import QAEvaluationRequest


async def get_evaluation_score(req_data: QAEvaluationRequest):
    try:
        qa_pairs = req_data.model_dump()
        qa_evaluator = QAEvaluator()
        score = await qa_evaluator.evaluate_qa_pairs(qa_pairs)
        return score
    except Exception as e:
        return {"error": str(e)}