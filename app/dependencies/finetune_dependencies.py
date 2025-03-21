from app.controllers.finetune_controller import FinetuneController
from app.repositories.finetune_repositories import FinetuneRepository


def get_finetune_controller():
    """
    Returns an instance of FinetuneController.

    Returns:
        FinetuneController: An instance of FinetuneController.
    """
    finetune_repo = FinetuneRepository()
    return FinetuneController(finetune_repo)


# Export the required function
__all__ = [
    "get_finetune_controller",
]