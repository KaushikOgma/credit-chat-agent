"""
This file contains constants used in the application.
"""

from enum import Enum


class DBCollections(Enum):
    USER = "user"
    LOG = "log"
    TRAIN_DATA = "train_data"
    TEST_DATA = "test_data"
    CHAT_HISTORY = "chat_history"
    METADATA = "metadata"

class RoutePrefix(Enum):
    """
    An enumeration of route prefixes used in the application.

    This enumeration defines a set of constants that represent different
    route prefixes used in the application.
    """

    MODULE = "/module"
    LOG = "/log"
    USER = "/user"
    METADATA = "/metadata"
    DATA_INGESTION = "/data_ingestion"
    FINETUNE = "/finetune"
    EVALUATION = "/eval"


class RouteTag(Enum):
    """
    An enumeration of route tags used in the application.

    This enumeration defines a set of constants that represent different
    route tags used in the application. The values defined in this enumeration
    can be used to categorize routes and provide additional metadata.
    """

    MODULE = "Modules"
    LOG = "Log"
    USER = "User"
    METADATA = "Metadata"
    DATA_INGESTION = "Data Ingestion"
    FINETUNE = "Finetune"
    EVALUATION = "Eval"
