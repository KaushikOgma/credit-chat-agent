import io
import os
import tempfile
from app.services.qa_generator import generate_question_and_answer
from app.schemas.module_schema import QAGenerateRequest


async def generate_qa(req_data: QAGenerateRequest):
    try:
        text = req_data.text
        # print(">>>>>>>>> ",text)
        qa = await generate_question_and_answer(text)
        return qa
    except Exception as e:
        return {"error": str(e)}