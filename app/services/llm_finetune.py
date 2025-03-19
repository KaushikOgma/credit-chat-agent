import os
import sys
sys.path.append(os.getcwd())
import asyncio
import aiofiles
import json
import re
import tempfile
from openai import AsyncOpenAI
from app.utils.config import settings
from app.utils.helpers.prompt_helper import PromptHelper

class OpenAIFineTuner:
    """
    A class for fine-tuning an OpenAI model using structured Q&A data.
    """
    def __init__(self):
        """
        Initializes the OpenAIFineTuner class with OpenAI client and model settings.
        """
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.BASE_MODEL_FOR_FINETUNE
        self.check_interval = 30  
        self.system_prompt = PromptHelper.jsonl_system_content_massage()[0].content

    @staticmethod
    def clean_question(question: str) -> str:
        """
        Cleans and formats a question by removing unwanted characters.
        """
        question = re.sub(r'^[\d]+[).\-*\s]*', '', question)  # Remove leading numbers
        question = re.sub(r'^[#*\s]+|[#*\s]+$', '', question)  # Trim special chars
        return question.strip()

    async def convert_to_jsonl(self, qa_data: list) -> list:
        """
        Converts structured Q&A data into JSONL format.
        """
        jsonl_data = []
        try:
            for qa in qa_data:
                cleaned_question = self.clean_question(qa["question"])
                formatted_entry = {
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": cleaned_question},
                        {"role": "assistant", "content": qa["answer"]}
                    ]
                }
                jsonl_data.append(json.dumps(formatted_entry, ensure_ascii=False))

            print("Q&A successfully converted to JSONL format in memory!")
            return jsonl_data

        except Exception as e:
            print(f"Error processing JSONL conversion: {e}")
            return []

    async def upload_jsonl_data(self, jsonl_data: list) -> str:
        """
        Saves JSONL data to a temporary file and uploads it to OpenAI.
        """
        try:
            async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as temp_file:
                await temp_file.write("\n".join(jsonl_data).encode("utf-8"))
                temp_file_path = temp_file.name

            response = await self.client.files.create(
                file=open(temp_file_path, "rb"),
                purpose="fine-tune"
            )
            print(f"File uploaded successfully! File ID: {response.id}")
            return response.id
        except Exception as e:
            print(f"Error uploading file: {e}")
            return ""

    async def fine_tune_model(self, file_id: str) -> str:
        """
        Initiates fine-tuning and monitors progress.
        """
        try:
            response = await self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=self.model
            )
            fine_tune_id = response.id
            print(f"Fine-tuning started! Job ID: {fine_tune_id}")

            while True:
                job_status = await self.client.fine_tuning.jobs.retrieve(fine_tune_id)
                status = job_status.status

                if status in ["succeeded", "failed"]:
                    break

                print(f"Fine-tuning in progress... (Status: {status})")
                await asyncio.sleep(self.check_interval)

            if status == "succeeded":
                model_id = job_status.fine_tuned_model
                print(f"Fine-tuning completed! Model ID: {model_id}")
                return model_id
            else:
                print("Fine-tuning failed. Check OpenAI logs.")
                return ""

        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            return ""


import asyncio

async def main():
    # Initialize the fine-tuner
    fine_tuner = OpenAIFineTuner()

    # Sample Q&A data for testing
    qa_data = [
        {"question": "1) What is OpenAI?", "answer": "OpenAI is an AI research and deployment company."},
        {"question": "2. How does GPT work?", "answer": "GPT uses deep learning to generate human-like text."}
    ]

    # Convert to JSONL format
    jsonl_data = await fine_tuner.convert_to_jsonl(qa_data)
    
    if not jsonl_data:
        print("JSONL conversion failed.")
        return

    # Upload JSONL data
    file_id = await fine_tuner.upload_jsonl_data(jsonl_data)

    if not file_id:
        print("File upload failed.")
        return

    # Fine-tune the model
    model_id = await fine_tuner.fine_tune_model(file_id)

    if model_id:
        print(f"Fine-tuning completed! New Model ID: {model_id}")
    else:
        print("Fine-tuning failed.")

# Run the test
if __name__ == "__main__":
    asyncio.run(main())
