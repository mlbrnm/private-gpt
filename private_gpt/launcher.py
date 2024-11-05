"""FastAPI app creation, logger configuration and main API routes."""

import logging

from fastapi import Depends, FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from injector import Injector
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.global_handlers import create_global_handler
from llama_index.core.settings import Settings as LlamaIndexSettings

from private_gpt.server.chat.chat_router import chat_router
from private_gpt.server.chunks.chunks_router import chunks_router
from private_gpt.server.completions.completions_router import completions_router
from private_gpt.server.embeddings.embeddings_router import embeddings_router
from private_gpt.server.health.health_router import health_router
from private_gpt.server.ingest.ingest_router import ingest_router
from private_gpt.server.recipes.summarize.summarize_router import summarize_router
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


def create_app(root_injector: Injector) -> FastAPI:

    # Start the API
    async def bind_injector_to_request(request: Request) -> None:
        request.state.injector = root_injector

    app = FastAPI(dependencies=[Depends(bind_injector_to_request)])

    app.include_router(completions_router)
    app.include_router(chat_router)
    app.include_router(chunks_router)
    app.include_router(ingest_router)
    app.include_router(summarize_router)
    app.include_router(embeddings_router)
    app.include_router(health_router)

    # Add LlamaIndex simple observability
    global_handler = create_global_handler("simple")
    if global_handler:
        LlamaIndexSettings.callback_manager = CallbackManager([global_handler])

    settings = root_injector.get(Settings)
    if settings.server.cors.enabled:
        logger.debug("Setting up CORS middleware")
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=settings.server.cors.allow_credentials,
            allow_origins=settings.server.cors.allow_origins,
            allow_origin_regex=settings.server.cors.allow_origin_regex,
            allow_methods=settings.server.cors.allow_methods,
            allow_headers=settings.server.cors.allow_headers,
        )

    @app.middleware("http")
    async def restrict_webui_access(request: Request, call_next):
        # Log the request and headers for debugging. Uncomment to enable.
        # logger.info(f"Processing request path: {request.url.path}")
        # logger.info(f"Request headers: {request.headers}")

        # Only apply the restriction for the WebUI root path and static files
        if request.url.path == "/" or request.url.path.startswith("/static"):
            # Check for the X-Forwarded-For header to determine if it's a proxy request
            if not request.headers.get("x-forwarded-for"):
                logger.warning("Direct access to WebUI detected, blocking request.")
                raise HTTPException(status_code=403, detail="Access to WebUI forbidden. Use :8002")
        
        # Ensure the next middleware is called properly and a valid response is returned
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error while processing request: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

    if settings.ui.enabled:
        logger.debug("Importing the UI module")
        try:
            from private_gpt.ui.ui import PrivateGptUi
        except ImportError as e:
            raise ImportError(
                "UI dependencies not found, install with `poetry install --extras ui`"
            ) from e

        ui = root_injector.get(PrivateGptUi)
        ui.mount_in_app(app, settings.ui.path)

    return app
