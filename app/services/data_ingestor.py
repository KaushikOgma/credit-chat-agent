import asyncio
import json
import os
import fitz
import pytesseract
from PIL import Image
from docx import Document
import whisper
from app.utils.config import settings

class DataIngestor:
    
    def extract_text(self, file_path, file_extension):
        try:
            if file_extension == "pdf":
                with fitz.open(file_path) as doc:
                    return "\n".join([page.get_text() for page in doc])

            elif file_extension in ["png", "jpg", "jpeg", "tiff", "bmp", "gif"]:
                image = Image.open(file_path)
                return pytesseract.image_to_string(image)

            elif file_extension == "docx":
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])

            elif file_extension in ["mp3", "wav", "m4a", "flac"]:
                model = whisper.load_model("base")
                result = model.transcribe(file_path)
                return result["text"]

            elif file_extension == "txt":
                with open(file_path, "r", encoding="utf-8") as txt_file:
                    return txt_file.read()

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return ""

    def ingest_files(self, file_path_list):
        extracted_texts = {}
        for curr_file_path in file_path_list:
            file_name = os.path.split(curr_file_path)[-1]
            file_extension = file_name.lower().split('.')[-1]
            text = self.extract_text(curr_file_path, file_extension)
            if text:
                extracted_texts[file_name] = text
        return extracted_texts




async def start_ingetion():
    # Initialize the fine-tuner
    data_ingestor = DataIngestor()

    
    # create a directory called input_data under uploads directory
    # and drop all testing files there

    target_folder = os.path.join(".","uploads","input_data")
    target_files = list(os.listdir(target_folder))
    extracted_text = data_ingestor.ingest_files(target_files)
    with open(os.path.join(settings.LOCAL_UPLOAD_LOCATION,'extracted_data.json'), 'w') as f:
        json.dump(extracted_text, f, indent=4)

if __name__ == "__main__":
    asyncio.run(start_ingetion())
