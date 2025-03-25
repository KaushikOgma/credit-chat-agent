from app.services.chat_service import ChatService
from app.controllers.chat_controller import ChatController
from app.repositories.evaluation_repositories import EvaluationRepository

def get_chat_controller():
    """
    Returns an instance of DataIngestionController.

    Returns:
        DataIngestionController: An instance of DataIngestionController.
    """
    eval_repo = EvaluationRepository()
    chat_service = ChatService()
    return ChatController(chat_service, eval_repo)

# Export the required function
__all__ = [
    "get_chat_controller",
]