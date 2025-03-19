import os
from langchain.schema import HumanMessage, SystemMessage
from app.utils.config import settings



def create_text_cleaning_conversation_messages(text):
    """
    Takes a piece of text and returns a list of messages designed to clean the text.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        list: A list of messages that set up the context for cleaning the text.
    """
    # Create a system message setting the context for the extraction task
    context_message = SystemMessage(content=settings.TEXT_CLEANING_SYSTEM_PROMPT)
    
    # Create a human message containing the input text
    input_text_message = HumanMessage(content=text)
    
    # Return the list of messages to be used in the extraction conversation
    return [context_message, input_text_message]


def create_question_extraction_conversation_messages(text):
    """
    Takes a piece of text and returns a list of messages designed to extract questions from the text.
    
    Args:
        text (str): The input text for which questions are to be extracted.
    
    Returns:
        list: A list of messages that set up the context for extracting questions.
    """
    # Create a system message setting the context for the extraction task
    context_message = SystemMessage(content=settings.QUESTION_EXTRACTION_SYSTEM_PROMPT)
    
    # Create a human message containing the input text
    input_text_message = HumanMessage(content=text)
    
    # Return the list of messages to be used in the extraction conversation
    return [context_message, input_text_message]


def create_answer_generation_conversation_messages(question, text):
    """
    Takes a question and a text and returns a list of messages designed to answer the question based on the text.
    
    Args:
        question (str): The question to be answered.
        text (str): The text containing information for answering the question.
    
    Returns:
        list: A list of messages that set up the context for answering the question.
    """
    # Create a system message setting the context for the answering task
    context_message = SystemMessage(content=settings.ANSWER_GENERATION_SYSTEM_PROMPT)
    
    # Create a human message containing the input text
    input_text_message = HumanMessage(content=text)
    
    # Create a human message containing the question to be answered
    input_question_message = HumanMessage(content=question)
    
    # Return the list of messages to be used in the answering conversation
    return [context_message, input_text_message, input_question_message]


def jsonl_system_content_massage(self):
    
    context_message = SystemMessage(content=settings.JSONL_SYSTEM_PROMPT)
    
    return [context_message]