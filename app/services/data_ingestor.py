import os
import fitz
import pytesseract
from PIL import Image
from docx import Document
from langchain.schema import Document as LCDocument
import whisper

def extract_text(file_path, file_extension):
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


def process_folder(input_folder):
    extracted_texts = {}
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        file_extension = filename.lower().split('.')[-1]
        text = extract_text(file_path, file_extension)
        if text:
            extracted_texts[filename] = text
    return extracted_texts
