from app.services.data_ingestor import DataIngestor
from app.controllers.data_ingestion_controller import DataIngestionController
from app.repositories.metadata_repositories import MetadataRepository


def get_data_ingestion_controller():
    """
    Returns an instance of DataIngestionController.

    Returns:
        DataIngestionController: An instance of DataIngestionController.
    """
    metadata_repo = MetadataRepository()
    data_ingestor = DataIngestor()
    return DataIngestionController(metadata_repo, data_ingestor)


# Export the required function
__all__ = [
    "get_data_ingestion_controller",
]