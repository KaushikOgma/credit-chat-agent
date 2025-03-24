from app.controllers.evaluation_controller import EvaluationController
from app.repositories.evaluation_repositories import EvaluationRepository
from app.services.qa_evaluator import QAEvaluator


def get_eval_controller():
    """
    Returns an instance of EvaluationController.

    Returns:
        EvaluationController: An instance of EvaluationController.
    """
    finetune_repo = EvaluationRepository()
    question_evaluator = QAEvaluator()
    return EvaluationController(finetune_repo, question_evaluator)


# Export the required function
__all__ = [
    "get_eval_controller",
]