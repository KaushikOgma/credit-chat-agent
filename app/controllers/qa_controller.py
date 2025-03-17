import io
import os
import tempfile
from app.services.qa_generator import QAGenerator
from app.schemas.module_schema import QAGenerateRequest


async def generate_qa(req_data: QAGenerateRequest):
    try:
        text = req_data.text
        # print(">>>>>>>>> ",text)
        qa_generator = QAGenerator()
        qa = await qa_generator.generate_question_and_answer(text)
        return qa
    except Exception as e:
        return {"error": str(e)}