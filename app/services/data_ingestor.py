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
                model = whisper.load_model("base")
                audio_data = BytesIO(file_content)
                result = model.transcribe(audio_data)
                return result["text"]

            elif file_extension == "txt":
                return file_content.decode("utf-8")

        except Exception as error:
            print(f"Error processing {file_name}: {error}")
            return ""

    async def ingest_files(self, files: list) -> dict:
        """Processes multiple files and extracts text."""
        extracted_texts = {}
        for file_name, file_content in files:
            text = await self.extract_text(file_name, file_content)
            if text:
                extracted_texts[file_name] = text
        return extracted_texts
    
    
    
    
    


async def start_ingestion():
    """Runs the data ingestion process on files inside the uploads/input_data directory."""
    data_ingestor = DataIngestor()

    # Define target folder
    target_folder = os.path.join(".", "uploads", "input_data")

    if not os.path.exists(target_folder):
        print(f"Directory {target_folder} does not exist.")
        return

    # Get file list
    target_files = os.listdir(target_folder)
    if not target_files:
        print("No files found in the input_data folder.")
        return

    # Process files
    extracted_text = await data_ingestor.ingest_files(target_files)

    # Save extracted data
    output_file = os.path.join(settings.LOCAL_UPLOAD_LOCATION, 'extracted_data.json')
    with open(output_file, 'w') as f:
        json.dump(extracted_text, f, indent=4)

    print(f"Extraction completed. Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(start_ingestion())
