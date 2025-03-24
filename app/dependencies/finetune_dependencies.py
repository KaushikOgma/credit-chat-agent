from app.controllers.finetune_controller import FinetuneController
from app.repositories.finetune_repositories import FinetuneRepository
from app.services.llm_finetune import OpenAIFineTuner


def get_finetune_controller():
    """
    Returns an instance of FinetuneController.

    Returns:
        FinetuneController: An instance of FinetuneController.
    """
    finetune_repo = FinetuneRepository()
    opeai_finetuner = OpenAIFineTuner()
    return FinetuneController(finetune_repo, opeai_finetuner)


# Export the required function
__all__ = [
    "get_finetune_controller",
]