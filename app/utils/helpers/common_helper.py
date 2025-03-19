import re
import unicodedata
import uuid


def preprocess_text(text) -> str:
    """
    Preprocesses the input text by lowercasing, removing extra spaces, and normalizing Unicode.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    try:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)                       # Remove extra spaces
        text = unicodedata.normalize("NFKC", text)             # Unicode normalization
        return text
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return text



def generate_uuid():
    """**Summary:**
    Generate and return a UUID (Universally Unique Identifier).
    """
    try:
        return str(uuid.uuid4())
    except Exception as error:
        # print("generate_uuid:: error - " + str(error))
        raise error

