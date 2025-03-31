from app.services.data_ingestor import DataIngestor
from app.services.qa_generator import QAGenerator
from app.services.qa_evaluator import QAEvaluator
from app.controllers.metadata_controller import MetadataController
from app.repositories.metadata_repositories import MetadataRepository
from app.repositories.finetune_repositories import FinetuneRepository
from app.repositories.evaluation_repositories import EvaluationRepository
from app.controllers.finetune_controller import FinetuneController
from app.controllers.evaluation_controller import EvaluationController
from app.repositories.model_data_repositories import ModelDataRepository
from app.services.llm_finetune import OpenAIFineTuner


def get_metadata_controller():
    """
    Returns an instance of MetadataController.

    Returns:
        MetadataController: An instance of MetadataController.
    """
    metadata_repo = MetadataRepository()
    finetune_repo = FinetuneRepository()
    eval_repo = EvaluationRepository()
    data_ingestor = DataIngestor()
    qa_generator = QAGenerator()
    qa_evaluator = QAEvaluator()
    opeai_finetuner = OpenAIFineTuner()
    model_data_repo = ModelDataRepository()
    finetune_controller = FinetuneController(finetune_repo, model_data_repo, opeai_finetuner)
    eval_controller = EvaluationController(eval_repo, model_data_repo, qa_evaluator)
    return MetadataController(metadata_repo, finetune_repo, eval_repo, finetune_controller, eval_controller, data_ingestor, qa_generator, qa_evaluator)


# Export the required function
__all__ = [
    "get_metadata_controller",
]