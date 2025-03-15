"""
This module defines the router for the application. The router is responsible for
including all the routes in the application.
"""

from fastapi import APIRouter
from app.routes import module_route
from app.utils.constants import RouteTag, RoutePrefix

# Initialize all routes
router = APIRouter()

# Include auth routes
router.include_router(
    module_route.router, prefix=RoutePrefix.MODULE.value, tags=[RouteTag.MODULE.value]
)