import asyncio
import os
import fitz
import pytesseract
from PIL import Image
from docx import Document
from langchain.schema import Document as LCDocument
import whisper

async def extract_text(file_path: str, file_extension: str) -> str:
    try:
        if file_extension == "pdf":
            return await asyncio.to_thread(extract_pdf_text, file_path)
        elif file_extension in {"png", "jpg", "jpeg", "tiff", "bmp", "gif"}:
            return await asyncio.to_thread(extract_image_text, file_path)
        elif file_extension == "docx":
            return await asyncio.to_thread(extract_docx_text, file_path)
        elif file_extension in {"mp3", "wav", "m4a", "flac"}:
            return await transcribe_audio(file_path)
        elif file_extension == "txt":
            return await asyncio.to_thread(extract_txt_text, file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return ""

async def extract_pdf_text(file_path: str) -> str:
    with fitz.open(file_path) as doc:
        return "\n".join([page.get_text() for page in doc])

async def extract_image_text(file_path: str) -> str:
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)

async def extract_docx_text(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

async def extract_txt_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as txt_file:
        return txt_file.read()

async def transcribe_audio(file_path: str) -> str:
    model = whisper.load_model("base")
    result = await asyncio.to_thread(model.transcribe, file_path)
    return result["text"]

async def process_folder(input_folder: str) -> dict:
    extracted_texts = {}
    tasks = []

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        file_extension = filename.lower().split('.')[-1]
        tasks.append(extract_text(file_path, file_extension))

    results = await asyncio.gather(*tasks)
    for filename, text in zip(os.listdir(input_folder), results):
        if text:
            extracted_texts[filename] = text

    return extracted_texts