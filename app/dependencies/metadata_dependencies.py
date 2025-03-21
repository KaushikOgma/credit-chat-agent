from app.controllers.metadata_controller import MetadataController
from app.repositories.metadata_repositories import MetadataRepository


def get_metadata_controller():
    """
    Returns an instance of MetadataController.

    Returns:
        MetadataController: An instance of MetadataController.
    """
    metadata_repo = MetadataRepository()
    return MetadataController(metadata_repo)


# Export the required function
__all__ = [
    "get_metadata_controller",
]