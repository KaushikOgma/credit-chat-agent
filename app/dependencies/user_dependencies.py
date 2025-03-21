from app.controllers.user_controller import UserController
from app.repositories.user_repositories import UserRepository


def get_user_controller():
    """
    Returns an instance of UserRepository.

    Returns:
        UserRepository: An instance of UserRepository.
    """
    user_repo = UserRepository()
    return UserController(user_repo)


# Export the required function
__all__ = [
    "get_user_controller",
]