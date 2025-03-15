"""
This file contains constants used in the application.
"""

from enum import Enum



class RoutePrefix(Enum):
    """
    An enumeration of route prefixes used in the application.

    This enumeration defines a set of constants that represent different
    route prefixes used in the application.
    """

    MODULE = "/module"


class RouteTag(Enum):
    """
    An enumeration of route tags used in the application.

    This enumeration defines a set of constants that represent different
    route tags used in the application. The values defined in this enumeration
    can be used to categorize routes and provide additional metadata.
    """

    MODULE = "Modules"
