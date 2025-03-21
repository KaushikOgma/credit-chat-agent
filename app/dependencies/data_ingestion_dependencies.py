from app.services.data_ingestor import DataIngestor
from app.services.qa_generator import QAGenerator
from app.controllers.data_ingestion_controller import DataIngestionController
from app.repositories.metadata_repositories import MetadataRepository
from app.repositories.finetune_repositories import FinetuneRepository
from app.repositories.evaluation_repositories import EvaluationRepository


def get_data_ingestion_controller():
    """
    Returns an instance of DataIngestionController.

    Returns:
        DataIngestionController: An instance of DataIngestionController.
    """
    metadata_repo = MetadataRepository()
    finetune_repo = FinetuneRepository()
    eval_repo = EvaluationRepository()
    data_ingestor = DataIngestor()
    qa_generator = QAGenerator()
    return DataIngestionController(metadata_repo, finetune_repo, eval_repo, data_ingestor, qa_generator)


# Export the required function
__all__ = [
    "get_data_ingestion_controller",
]