import asyncio
import json
import os
import sys
sys.path.append(os.getcwd())
import fitz
import pytesseract
from PIL import Image
from docx import Document
import whisper
from io import BytesIO
from app.utils.config import settings
from app.utils.logger import setup_logger

logger = setup_logger()

class DataIngestor:
    def __init__(self):
        self.service_name = "data_ingestor"
        self.whisper_model = whisper.load_model("base")  # Load once for efficiency

    async def extract_text(self, file_name: str, file_content: bytes) -> str:
        """Extracts text from various file formats."""
        try:
            file_extension = file_name.split(".")[-1].lower()

            if file_extension == "pdf":
                pdf = fitz.open(stream=file_content, filetype="pdf")
                return "\n".join([page.get_text() for page in pdf])

            elif file_extension in ["png", "jpg", "jpeg", "tiff", "bmp", "gif"]:
                image = Image.open(BytesIO(file_content))
                return pytesseract.image_to_string(image)

            elif file_extension == "docx":
                doc = Document(BytesIO(file_content))
                return "\n".join([para.text for para in doc.paragraphs])

            elif file_extension in ["mp3", "wav", "m4a", "flac"]:
                audio_file = BytesIO(file_content)
                result = self.whisper_model.transcribe(audio_file)
                return result["text"]

            elif file_extension == "txt":
                return file_content.decode("utf-8")

        except Exception as error:
            logger.error(f"Error processing {file_name}: {error}")
            return ""

    async def ingest_files(self, files: list) -> dict:
        """Processes multiple files and extracts text."""
        extracted_texts = {}
        for file in files:
            file_name = file.filename
            file_content = await file.read()
            text = await self.extract_text(file_name, file_content)
            if text:
                extracted_texts[file_name] = text
        return extracted_texts
    
    def ingest_file_from_path(self, file_path_list):
        extracted_texts = {}
        try:
            for curr_file_path in file_path_list:
                file_name = os.path.split(curr_file_path)[-1]
                file_extension = file_name.lower().split('.')[-1]
                text = self.extract_text(curr_file_path, file_extension)
                if text:
                    extracted_texts[file_name] = text
            return extracted_texts
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return extracted_texts



async def start_ingetion():
    # Initialize the fine-tuner
    data_ingestor = DataIngestor()

    
    # create a directory called input_data under uploads directory
    # and drop all testing files there

    target_folder = os.path.join(".","uploads","input_data")
    target_files = list(os.listdir(target_folder))
    extracted_text = data_ingestor.ingest_file_from_path(target_files)
    with open(os.path.join(settings.LOCAL_UPLOAD_LOCATION,'extracted_data.json'), 'w') as f:
        json.dump(extracted_text, f, indent=4)

if __name__ == "__main__":
    asyncio.run(start_ingetion())
