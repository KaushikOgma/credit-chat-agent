"""
This module defines the router for the application. The router is responsible for
including all the routes in the application.
"""

from fastapi import APIRouter
from app.routes import log_route
from app.routes import user_route
from app.routes import metadata_route
from app.routes import data_ingestion_route
from app.routes import finetune_route
from app.routes import evaluation_route
from app.routes import chat_route
from app.utils.constants import RouteTag, RoutePrefix

# Initialize all routes
router = APIRouter()

router.include_router(
    log_route.router, prefix=RoutePrefix.LOG.value, tags=[RouteTag.LOG.value]
)
router.include_router(
    user_route.router, prefix=RoutePrefix.USER.value, tags=[RouteTag.USER.value]
)
router.include_router(
    data_ingestion_route.router, prefix=RoutePrefix.DATA_INGESTION.value, tags=[RouteTag.DATA_INGESTION.value]
)
router.include_router(
    metadata_route.router, prefix=RoutePrefix.METADATA.value, tags=[RouteTag.METADATA.value]
)
router.include_router(
    finetune_route.router, prefix=RoutePrefix.FINETUNE.value, tags=[RouteTag.FINETUNE.value]
)
router.include_router(
    evaluation_route.router, prefix=RoutePrefix.EVALUATION.value, tags=[RouteTag.EVALUATION.value]
)
router.include_router(
    chat_route.router, prefix=RoutePrefix.CHAT.value, tags=[RouteTag.CHAT.value]
)