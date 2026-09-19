from contextlib import asynccontextmanager
import logging
import threading

from fastapi import FastAPI

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config.logs_file import setup_logging

from routes.ingestion_route import (
    router as ingestion_router,
    limiter
)

from routes.chat_route import (
    router as chat_router
)


# =========================================================
# LOGGING
# =========================================================

setup_logging()

logger = logging.getLogger(__name__)


# =========================================================
# CREATE APPLICATION
# =========================================================

def create_app() -> FastAPI:

    logger.info(
        "FastAPI application starting"
    )

    # -----------------------------------------------------
    # Application lifespan
    # -----------------------------------------------------

    @asynccontextmanager
    async def lifespan(app: FastAPI):

        logger.info(
            "Application startup"
        )

        # Start Corpora filesystem watcher
        start_corpora_watcher()

        yield

        # -------------------------------------------------
        # Shutdown
        # -------------------------------------------------

        logger.info(
            "Application shutdown"
        )

        # If your CorporaWatcher has a stop method,
        # call it here.
        #
        # stop_corpora_watcher()

        logger.info(
            "Application shutdown completed"
        )

    # -----------------------------------------------------
    # FastAPI application
    # -----------------------------------------------------

    app = FastAPI(
        title="Agentic RAG API",
        version="1.0.0",
        lifespan=lifespan
    )

    # =====================================================
    # RATE LIMITING
    # =====================================================

    app.state.limiter = limiter

    app.add_exception_handler(
        RateLimitExceeded,
        _rate_limit_exceeded_handler
    )

    # =====================================================
    # ROUTES
    # =====================================================

    include_routes(app)

    return app


# =========================================================
# ROUTE REGISTRATION
# =========================================================

def include_routes(
    app: FastAPI
):

    # -----------------------------------------------------
    # Ingestion routes
    # -----------------------------------------------------

    app.include_router(
        ingestion_router,
        prefix="/ingest",
        tags=["Ingestion"]
    )

    # -----------------------------------------------------
    # Chat routes
    # -----------------------------------------------------

    app.include_router(
        chat_router,
        prefix="/chat",
        tags=["Chat"]
    )


# =========================================================
# CORPORA WATCHER
# =========================================================

def start_corpora_watcher():

    try:

        from watcher.corpora_watcher import (
            CorporaWatcher
        )

    except ImportError as exc:

        logger.warning(
            "Watchdog is not installed; "
            "Corpora watcher disabled. "
            "Install it with "
            "'pip install watchdog'. %s",
            exc
        )

        return

    try:

        watcher = CorporaWatcher(
            "corpora"
        )

        thread = threading.Thread(
            target=watcher.start,
            daemon=True,
            name="CorporaWatcher"
        )

        thread.start()

        logger.info(
            "Corpora watcher started"
        )

    except Exception:

        logger.exception(
            "Failed to start Corpora watcher"
        )


# =========================================================
# APPLICATION INSTANCE
# =========================================================

app = create_app()