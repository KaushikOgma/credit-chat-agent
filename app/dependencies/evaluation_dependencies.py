from app.controllers.evaluation_controller import EvaluationController
from app.repositories.evaluation_repositories import EvaluationRepository


def get_eval_controller():
    """
    Returns an instance of EvaluationController.

    Returns:
        EvaluationController: An instance of EvaluationController.
    """
    finetune_repo = EvaluationRepository()
    return EvaluationController(finetune_repo)


# Export the required function
__all__ = [
    "get_eval_controller",
]