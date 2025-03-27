from app.services.chat_service import ChatService
from app.controllers.chat_controller import ChatController
from app.repositories.model_data_repositories import ModelDataRepository

def get_chat_controller():
    """
    Returns an instance of DataIngestionController.

    Returns:
        DataIngestionController: An instance of DataIngestionController.
    """
    model_data_repo = ModelDataRepository()
    chat_service = ChatService()
    return ChatController(chat_service, model_data_repo)

# Export the required function
__all__ = [
    "get_chat_controller",
]