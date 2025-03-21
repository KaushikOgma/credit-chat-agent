from app.services.data_ingestor import DataIngestor
from app.services.qa_generator import QAGenerator
from app.controllers.metadata_controller import MetadataController
from app.repositories.metadata_repositories import MetadataRepository
from app.repositories.finetune_repositories import FinetuneRepository
from app.repositories.evaluation_repositories import EvaluationRepository


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
    return MetadataController(metadata_repo, finetune_repo, eval_repo, data_ingestor, qa_generator)


# Export the required function
__all__ = [
    "get_metadata_controller",
]