"""
This module contains all configurations that are used in the project.

Usage:
    from app.utils.config import Settings

    settings = Settings()

    print(settings.PROJECT_NAME)
"""

import os
from pydantic import Field, model_validator
import yaml
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

ENV_FILE = f".env.{os.getenv('PROJECT_ENV', 'dev')}"
# Load environment variables from .env file
load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    """
    This class contains all configurations that are used in the project.

    It loads environment variables from a .env file and provides a way to access
    these variables in a typed and structured way.

    Attributes:
        PROJECT_NAME (str): The name of the project.
        PROJECT_VERSION (str): The version of the project.
        APP_PORT (int): The port number of the application.
        PROJECT_ROOT_PATH (str): The root path of the project.
        PROJECT_ENV (str): The Python environment.
        ...
    """

    # Server Configuration
    PROJECT_NAME: str = Field(..., env="PROJECT_NAME")
    PROJECT_VERSION: str = Field(..., env="PROJECT_VERSION")
    APP_PORT: int = Field(..., env="APP_PORT")
    PROJECT_ROOT_PATH: str = Field(..., env="PROJECT_ROOT_PATH")
    PROJECT_ENV: str = Field(..., env="PROJECT_ENV")
    MODULE: str = Field(..., env="MODULE")
    MAX_THREADS: int = Field(..., env="MAX_THREADS")
    MAX_PROCESSES: int = Field(..., env="MAX_PROCESSES")
    APP_TIMEZONE: str = Field(..., env="APP_TIMEZONE")
    ACCEPTED_DATE_TIME_STRING: str = Field(..., env="ACCEPTED_DATE_TIME_STRING")
    LOG_FILE: str = Field(..., env="LOG_FILE")
    LOCAL_UPLOAD_LOCATION: str = Field(..., env="LOCAL_UPLOAD_LOCATION")

    # # JWT Configuration
    # ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(..., env="ACCESS_TOKEN_EXPIRE_MINUTES")
    # REFRESH_TOKEN_EXPIRE_DAYS: int = Field(..., env="REFRESH_TOKEN_EXPIRE_DAYS")
    # ENCRYPTION_KEY: str = Field(..., env="ENCRYPTION_KEY")

    # Extra Settings
    # SEEDING: str = Field(..., env="SEEDING")
    # SQL_LOG: str = Field(..., env="SQL_LOG")

    # # s3 settings
    # S3_BUCKET: str = Field(..., env="S3_BUCKET")
    # AWS_ACCESS_KEY_ID: str = Field(..., env="AWS_ACCESS_KEY_ID")
    # AWS_SECRET_ACCESS_KEY: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    # AWS_REGION: str = Field(..., env="AWS_REGION")


    # Model Configuration
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    BASE_MODEL: str = Field(..., env="BASE_MODEL")
    ADVANCE_MODEL: str = Field(..., env="ADVANCE_MODEL")
    BASE_MODEL_FOR_FINETUNE: str = Field(..., env="BASE_MODEL_FOR_FINETUNE")
    BASE_AUDIO_MODEL: str = Field(..., env="BASE_AUDIO_MODEL")
    BASE_MODEL_TOKEN_LIMIT: int = Field(..., env="BASE_MODEL_TOKEN_LIMIT")
    BASE_MODEL_TOKENS_PER_MESSAGE: int = Field(..., env="BASE_MODEL_TOKENS_PER_MESSAGE")
    PROMPT_FILE: str = Field(..., env="PROMPT_FILE")
    AVERAGE_QUESTION_TOKEN_SIZE: int = Field(..., env="AVERAGE_QUESTION_TOKEN_SIZE")
    AVERAGE_ANSWER_TOKEN_SIZE: int = Field(..., env="AVERAGE_ANSWER_TOKEN_SIZE")
    AVERAGE_QUESTION_TEXT_RATIO: float = Field(..., env="AVERAGE_QUESTION_TEXT_RATIO")

    # DB Configuration
    PINECONE_API_KEY: str = Field(..., env="PINECONE_API_KEY")
    VECTOR_DB_NAME: str = Field(..., env="VECTOR_DB_NAME")
    VECTOR_DIMENSION: int = Field(..., env="VECTOR_DIMENSION")
    EMBEDDING_MODEL_NAME: str = Field(..., env="EMBEDDING_MODEL_NAME")

    DB_NAME: str = Field(..., env="DB_NAME")
    DB_PROTOCOL: str = Field(..., env="DB_PROTOCOL")
    DB_HOST: str = Field(..., env="DB_HOST")
    DB_PORT: str = Field(..., env="DB_PORT")
    DB_USER: str = Field(..., env="DB_USER")
    DB_PASSWORD: str = Field(..., env="DB_PASSWORD")
    SEEDING: str = Field(..., env="SEEDING")
    VERBOSE: str = Field(..., env="VERBOSE")

    # Array Configuration
    ARRAY_APP_KEY: str = Field(..., env="ARRAY_APP_KEY")
    ARRAY_SERVER_TOKEN: str = Field(..., env="ARRAY_SERVER_TOKEN")
    ARRAY_BASE_URL: str = Field(..., env="ARRAY_BASE_URL")
    
    #Chat Limit
    CHAT_HISTORY_LIMIT: int = Field(..., env="CHAT_HISTORY_LIMIT")

    @model_validator(mode="before")
    def load_yaml_values(cls, values):
        yaml_file_path = values.get("PROMPT_FILE")
        if yaml_file_path:
            try:
                with open(yaml_file_path, "r") as file:
                    yaml_data = yaml.safe_load(file) or {}
                    values.update(yaml_data)
            except FileNotFoundError:
                print(f"Warning: YAML file not found at {yaml_file_path}")
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file: {e}")
        return values

    class Config:
        """
        Configuration settings for the application.

        This class defines the configuration settings for the application, including
        the environment file, encoding, and extra settings.
        """

        # Assuming your .env file is in the same directory as your config.py
        env_file = ENV_FILE
        env_file_encoding = "utf-8"
        extra = "allow"

settings = Settings()