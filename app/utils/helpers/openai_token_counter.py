import tiktoken
import os
from app.utils.helpers.prompt_helper import create_question_extraction_conversation_messages, create_answer_generation_conversation_messages
from app.utils.config import settings

# properties of the model used
model_tokens_per_name = -1  # Same behavior as GPT-3.5-turbo
# parameters
padding_token_count = 16  # Same logic applies to ensure we stay within the limit
# encoder used to turn text into token
encoding = tiktoken.encoding_for_model(settings.BASE_MODEL)


def count_tokens_text(text):
    """
    Counts the number of tokens used to encode a given text.
    
    Args:
        text (str): The input text to be tokenized.
        
    Returns:
        int: The number of tokens in the encoded text.
    """
    try:
        encoded_text = encoding.encode(text)
        return len(encoded_text)
    except Exception as e:
        print(f"Error counting tokens for text: {e}")
        return len(text.split())


def count_tokens_messages(messages):
    """
    Counts the number of tokens needed to encode a list of messages.
    
    Args:
        messages (list): A list of message objects to be tokenized.
        
    Returns:
        int: The total number of tokens required to encode the messages.
        
    Adapted from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    total_tokens = 0
    try:
        for message in messages:
            total_tokens += settings.BASE_MODEL_TOKENS_PER_MESSAGE
            total_tokens += count_tokens_text(message.content)
            total_tokens += model_tokens_per_name
        total_tokens += settings.BASE_MODEL_TOKENS_PER_MESSAGE # every reply is primed with <|start|>assistant<|message|>
        return total_tokens
    except Exception as e:
        print(f"Error counting tokens for messages: {e}")
        return total_tokens


def get_available_tokens(messages_token_count):
    """
    Calculates the number of tokens that can be requested from the model.
    
    Args:
        messages_token_count (int): The total number of tokens used by the messages.
        
    Returns:
        int: The number of tokens available for the model request.
    """
    try:
        adjusted_token_limit = settings.BASE_MODEL_TOKEN_LIMIT - padding_token_count  # Avoid requesting the exact token limit
        available_tokens = adjusted_token_limit - messages_token_count
        return available_tokens
    except Exception as e:
        print(f"Error calculating available tokens: {e}")
        return 0




def estimate_question_extraction_conversation_tokens(text_token_count):
    """
    Estimates the total number of tokens needed for the extraction conversation.
    
    Args:
        text_token_count (int): The total number of tokens in the input text.
        
    Returns:
        float: The estimated total number of tokens needed for the extraction conversation.
    """
    try:
        empty_question_extraction_messages_token_count = count_tokens_messages(create_question_extraction_conversation_messages(text=''))

        # Calculate the upper bound of output tokens based on the input text tokens
        # The upper bound is the maximum of an average question size or a number of questions
        # proportional to the text length, plus half an average question worth of padding
        upper_bound_output_size = max(
            settings.AVERAGE_QUESTION_TOKEN_SIZE, 
            text_token_count * settings.AVERAGE_QUESTION_TEXT_RATIO
        ) + settings.AVERAGE_QUESTION_TOKEN_SIZE / 2

        # The total estimated token count includes the extraction messages,
        # input text tokens, and the calculated upper bound output size
        estimated_token_count = (
            empty_question_extraction_messages_token_count + 
            text_token_count + 
            upper_bound_output_size
        )
        return estimated_token_count
    except Exception as e:
        print(f"Error estimating question extraction conversation tokens: {e}")
        return 0


def estimate_answer_generation_conversation_tokens(text_token_count):
    """
    Estimates the total number of tokens needed for the answering conversation.
    
    Args:
        text_token_count (int): The total number of tokens in the input text.
        
    Returns:
        float: The estimated total number of tokens needed for the answering conversation.
    """
    try:
        empty_answer_generation_messages_token_count = count_tokens_messages(create_answer_generation_conversation_messages(question='', text=''))
        # Calculate the upper bound of question tokens: one question plus half a question worth of padding
        upper_bound_question_size = settings.AVERAGE_QUESTION_TOKEN_SIZE * 1.5
        
        # Calculate the upper bound of answer tokens: one answer plus half an answer worth of padding
        upper_bound_answer_size = settings.AVERAGE_ANSWER_TOKEN_SIZE * 1.5
        
        # The total estimated token count includes the answering messages tokens,
        # input text tokens, and the calculated upper bound question and answer sizes
        estimated_token_count = (
            empty_answer_generation_messages_token_count + 
            text_token_count + 
            upper_bound_question_size + 
            upper_bound_answer_size
        )
        
        return estimated_token_count
    except Exception as e:
        print(f"Error estimating answer generation conversation tokens: {e}")
        return 0

def are_tokens_available_for_both_conversations(text_token_count):
    """
    Checks if there are enough tokens available to get an answer from the model for both extraction and answering.
    
    Args:
        text_token_count (int): The total number of tokens in the input text.
    
    Returns:
        bool: True if there are enough tokens available, False otherwise.
    """
    try:
        # Calculate tokens needed for extraction conversation
        tokens_needed_question_extraction = estimate_question_extraction_conversation_tokens(text_token_count)
        tokens_available_question_extraction = get_available_tokens(tokens_needed_question_extraction)
        
        # Calculate tokens needed for answering conversation
        tokens_needed_answer_generation = estimate_answer_generation_conversation_tokens(text_token_count)
        tokens_available_answer_generation = get_available_tokens(tokens_needed_answer_generation)
        
        # Check if there are enough tokens available for both extraction and answering
        return (tokens_available_question_extraction > 0) and (tokens_available_answer_generation > 0)
    except Exception as e:
        print(f"Error checking token availability for both conversations: {e}")
        return False
