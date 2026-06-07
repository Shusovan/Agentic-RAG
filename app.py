from contextlib import asynccontextmanager
import logging
import threading

from fastapi import FastAPI

from config.logs_file import setup_logging
from routes.ingestion_route import router as ingestion_router
from routes.chat_route import router as chat_router


setup_logging()

logger = logging.getLogger(__name__)


# def create_app() -> FastAPI:

    # logger.info("FastAPI app starting")

    # app = FastAPI(title="Agentic RAG API", version="1.0.0")

    # include_routes(app)

    # @app.on_event("startup")
    # async def startup_event():
        # start_corpora_watcher()

    # return app

def create_app() -> FastAPI:

    logger.info("FastAPI app starting")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup logic
        start_corpora_watcher()
        yield
        # Optional shutdown logic
        # stop_corpora_watcher()

    app = FastAPI(title="Agentic RAG API", version="1.0.0", lifespan=lifespan)

    include_routes(app)

    return app


def include_routes(app: FastAPI):
    app.include_router(ingestion_router, prefix="/ingest", tags=["Ingestion"])

    app.include_router(chat_router, prefix="/chat", tags=["Chat"])


def start_corpora_watcher():
    try:
        from watcher.corpora_watcher import CorporaWatcher
    except ImportError as exc:
        logger.warning(
            "Watchdog is not installed; corpora watcher disabled. Install it with 'pip install watchdog'. %s",
            exc,
        )
        return

    watcher = CorporaWatcher("corpora")
    thread = threading.Thread(target=watcher.start, daemon=True)
    thread.start()


app = create_app()