from app.services.data_ingestor import DataIngestor
from app.services.qa_generator import QAGenerator
from app.controllers.data_ingestion_controller import DataIngestionController
from app.repositories.metadata_repositories import MetadataRepository
from app.repositories.finetune_repositories import FinetuneRepository
from app.repositories.evaluation_repositories import EvaluationRepository
from app.controllers.metadata_controller import MetadataController
from app.services.qa_evaluator import QAEvaluator
from app.controllers.finetune_controller import FinetuneController
from app.controllers.evaluation_controller import EvaluationController
from app.services.llm_finetune import OpenAIFineTuner

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
    qa_evaluator = QAEvaluator()
    opeai_finetuner = OpenAIFineTuner()
    finetune_controller = FinetuneController(finetune_repo, opeai_finetuner)
    eval_controller = EvaluationController(eval_repo, qa_evaluator)
    matadata_controller = MetadataController(
        metadata_repo,
        finetune_repo,
        eval_repo,
        finetune_controller,
        eval_controller,
        data_ingestor,
        qa_generator,
        qa_evaluator
    )
    return DataIngestionController(metadata_repo, finetune_repo, eval_repo, finetune_controller, eval_controller, matadata_controller, data_ingestor, qa_generator, qa_evaluator)

# Export the required function
__all__ = [
    "get_data_ingestion_controller",
]