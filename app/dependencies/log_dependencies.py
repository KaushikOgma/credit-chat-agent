from app.controllers.log_controller import LogController
from app.repositories.log_repositories import LogRepository


def get_log_controller():
    """
    Returns an instance of LogController.

    Returns:
        LogController: An instance of LogController.
    """
    log_repo = LogRepository()
    return LogController(log_repo)


# Export the required function
__all__ = [
    "get_log_controller",
]