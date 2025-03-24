import asyncio
import json
import os
import sys
import numpy as np
import tempfile

from tqdm import tqdm
sys.path.append(os.getcwd())
import fitz
import pytesseract
from PIL import Image
from docx import Document
from fastapi import UploadFile
from pathlib import Path
from openai import OpenAI
from io import BytesIO
from app.utils.config import settings
from app.utils.logger import setup_logger
from pydub import AudioSegment

logger = setup_logger()


class DataIngestor:
    def __init__(self):
        self.service_name = "data_ingestor"
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.BASE_AUDIO_MODEL


    async def transcribe_audio_in_chunks(self, file_path, chunk_size=24000000):
        text = ""
        try:
            audio = AudioSegment.from_file(file_path)
            chunk_length = (chunk_size / len(audio.raw_data)) * len(audio)
            for i in range(0, len(audio), int(chunk_length)):
                chunk = audio[i:i + int(chunk_length)]
                temp_file = Path("temp_chunk.mp3")
                chunk.export(temp_file, format="mp3")

                try:
                    with temp_file.open("rb") as audio_file:
                        response = self.client.audio.transcriptions.create(
                            model=self.model, 
                            file=audio_file
                        )
                        text += response.text
                finally:
                    temp_file.unlink()
            return text
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return text
        

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
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_audio:
                    temp_audio.write(file_content)
                    temp_audio_path = temp_audio.name
                file_size = Path(temp_audio_path).stat().st_size
                try:
                    if file_size < 24000000:
                        # Use OpenAI Whisper API
                        with open(temp_audio_path, "rb") as audio_file:
                            response = self.client.audio.transcriptions.create(
                                model=self.model, 
                                file=audio_file
                            )
                        return response.text  
                    else:
                        return await self.transcribe_audio_in_chunks(temp_audio_path)
                finally:
                    os.remove(temp_audio_path)
                    
            elif file_extension == "txt":
                return file_content.decode("utf-8")
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return ""


    async def ingest_files(self, files: list[UploadFile]) -> dict:
        extracted_texts = {}
        try:
            for file in tqdm(files, desc="File reading in progress"):
                try:
                    file_content = await file.read()  # Read the file content asynchronously
                    text = await self.extract_text(file.filename, file_content)
                    content_type = file.content_type
                    file_name = file.filename
                    if text:
                        extracted_texts[file_name] = {"content_type":content_type, "content": text}
                except Exception as e:
                    logger.exception(f"Failed to process {file.filename}: {e}")

            return extracted_texts
        except Exception as error:
            logger.exception(error, extra={"moduleName": settings.MODULE, "serviceName": self.service_name})
            return extracted_texts
    
    def ingest_file_from_path(self, file_path_list):
        extracted_texts = {}
        try:
            for curr_file_path in file_path_list:
                file_name = os.path.split(curr_file_path)[-1]
                file_extension = file_name.lower().split('.')[-1]
                text = self.extract_text(curr_file_path, file_extension)
                if text:
                    extracted_texts[file_name] = {"content_type":file_extension, "content": text}
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
