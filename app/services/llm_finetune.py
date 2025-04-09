import os
import sys
from typing import List
sys.path.append(os.getcwd())
import asyncio
import aiofiles
import json
import re
import tempfile
from openai import AsyncOpenAI
from app.utils.config import settings
from app.utils.helpers.prompt_helper import finetune_system_content_massage
import asyncio
from app.utils.logger import setup_logger
logger = setup_logger()

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
        self.system_prompt = finetune_system_content_massage()
        self.service_name = "model_finetune_service"

    def clean_question(self, question: str) -> str:
        """
        Cleans and formats a question by removing unwanted characters.
        """
        try:
            question = re.sub(r'^[\d]+[).\-*\s]*', '', question)  # Remove leading numbers
            question = re.sub(r'^[#*\s]+|[#*\s]+$', '', question)  # Trim special chars
            return question.strip()
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return question

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
            logger.info(f"Q&A successfully converted to JSONL format in memory!", extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return jsonl_data
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
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
            logger.info(f"File uploaded successfully! File ID: {response.id}", extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return response.id
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return ""

    async def fine_tune_model(self, file_id: str) -> str:
        """
        Initiates fine-tuning and monitors progress.
        """
        job_id = None
        model_id = None
        try:
            response = await self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=self.model
            )
            job_id = response.id
            logger.info(f"Fine-tuning started! Job ID: {job_id}", extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            while True:
                job_status = await self.client.fine_tuning.jobs.retrieve(job_id)
                status = job_status.status

                if status in ["succeeded", "failed"]:
                    break
                logger.info(f"Fine-tuning in progress... (Status: {status})", extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
                await asyncio.sleep(self.check_interval)
            if status == "succeeded":
                model_id = job_status.fine_tuned_model
                logger.info(f"Fine-tuning completed! Model ID: {model_id}", extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
                return job_id, model_id
            else:
                logger.info(f"Fine-tuning Failed! Job ID: {job_id}", extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
                return job_id, model_id
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return job_id, model_id
        

    async def get_training_params(self, job_id: str):
        params = None
        try:
            response = await self.client.fine_tuning.jobs.retrieve(fine_tuning_job_id=job_id)
            params = response.model_dump()
            # params = {
            #     "fine_tuned_model": response.fine_tuned_model,
            #     "base_model": response.model,
            #     "training_file": response.training_file,
            #     "trained_tokens": response.trained_tokens,
            #     "fine_tuning_hyperparameters": {
            #         "batch_size": response.hyperparameters.batch_size,
            #         "learning_rate_multiplier": response.hyperparameters.learning_rate_multiplier,
            #         "n_epochs": response.hyperparameters.n_epochs
            #     },
            #     "training_method": response.method.type,
            #     "job_status": response.status,
            #     "result_files": response.result_files
            # }
            return params
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return params


    async def start_finetune(self, qa_data: List[dict]):
        train_info = {
            "file_id": None,
            "job_id": None,
            "model_id": None,
            "params": None,
        }
        try:
            # Convert to JSONL format
            jsonl_data = await self.convert_to_jsonl(qa_data)
            if jsonl_data:
                # Upload JSONL data
                file_id = await self.upload_jsonl_data(jsonl_data)
                if file_id:
                    train_info["file_id"] = file_id
                    # Fine-tune the model
                    job_id, model_id = await self.fine_tune_model(file_id)
                    train_info["job_id"] = job_id
                    train_info["model_id"] = model_id
                    if model_id:
                        train_info["params"] = await self.get_training_params(job_id)
                else:
                    logger.exception(f"JSONL file Upload failed.", extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            else:
                logger.exception(f"JSONL conversion failed.", extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return train_info
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return train_info




async def start_training():
    # Initialize the fine-tuner
    fine_tuner = OpenAIFineTuner()

    # Sample Q&A data for testing
    qa_data = [
        {"question": "1) What is OpenAI?", "answer": "OpenAI is an AI research and deployment company."},
        {"question": "2. How does GPT work?", "answer": "GPT uses deep learning to generate human-like text."},
        {"question": "3) What is fine-tuning in machine learning?", "answer": "Fine-tuning is adjusting a pre-trained model with additional data to improve its performance on a specific task."},
        {"question": "4) What is the difference between AI and ML?", "answer": "AI is a broad concept of machines performing tasks intelligently, while ML is a subset where machines learn from data."},
        {"question": "5) How does reinforcement learning work?", "answer": "Reinforcement learning trains an agent by rewarding desired behaviors and penalizing undesired ones."},
        {"question": "6) What is the role of transformers in NLP?", "answer": "Transformers help process sequential data in parallel, improving efficiency and accuracy in NLP tasks."},
        {"question": "7) What is a neural network?", "answer": "A neural network is a computational model inspired by the human brain, used in deep learning to process complex patterns."},
        {"question": "8) How does backpropagation work?", "answer": "Backpropagation adjusts model weights by calculating and propagating the error backward through the network."},
        {"question": "9) What is the difference between supervised and unsupervised learning?", "answer": "Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data."},
        {"question": "10) What is a large language model (LLM)?", "answer": "An LLM is an AI model trained on massive text data to understand and generate human-like language."},
        {"question": "11) How is ChatGPT trained?", "answer": "ChatGPT is trained using reinforcement learning from human feedback (RLHF) and fine-tuned on conversation datasets."},
        {"question": "12) What is tokenization in NLP?", "answer": "Tokenization is the process of breaking text into words or subwords for analysis by NLP models."},
        {"question": "13) What is the role of embeddings in NLP?", "answer": "Embeddings convert words into numerical representations to help models understand meaning and context."},
        {"question": "14) How does a chatbot work?", "answer": "A chatbot processes user input using NLP techniques and responds based on pre-trained models or logic."},
        {"question": "15) What is zero-shot learning?", "answer": "Zero-shot learning enables AI models to make predictions on unseen tasks without additional training."},
        {"question": "16) What is a prompt in AI models?", "answer": "A prompt is an input or instruction given to an AI model to generate a response or perform a task."},
        {"question": "17) What are attention mechanisms in transformers?", "answer": "Attention mechanisms allow models to focus on relevant parts of the input, improving efficiency in NLP tasks."},
        {"question": "18) What is a fine-tuned model?", "answer": "A fine-tuned model is a pre-trained AI model further trained on a specific dataset for a specialized task."},
        {"question": "19) What is transfer learning?", "answer": "Transfer learning uses knowledge from a pre-trained model to improve performance on a new but related task."},
        {"question": "20) What is overfitting in machine learning?", "answer": "Overfitting occurs when a model learns noise from training data, reducing its generalization to new data."}
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
    job_id, model_id = await fine_tuner.fine_tune_model(file_id)

    if model_id:
        print(f"Fine-tuning completed with Job ID: {job_id}, New Model ID: {model_id}")
    else:
        print("Fine-tuning failed.")

# Run the test
if __name__ == "__main__":
    asyncio.run(start_training())
