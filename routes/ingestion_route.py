import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from langchain_core.documents import Document

from services.file_service import FileService
from ingestion import ingestion_pipeline
from schemas.ingestion_schema import TextIngestionRequest, WebIngestionRequest

from config.vector_dependency import vector_store, embedding_pipeline, bm25_store


router = APIRouter()

logger = logging.getLogger(__name__)

file_service = FileService()


pipeline = ingestion_pipeline.IngestionPipeline(vector_store=vector_store, bm25_store=bm25_store, embedding_pipeline=embedding_pipeline)


@router.post("/text")
def ingest_text(request: TextIngestionRequest):
    """
    Directly ingest raw text into the RAG pipeline.
    """

    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Text too small for ingestion")

    try:

        # Convert text → Document
        document = Document(page_content=text, metadata={"source": "text_input"})

        # Run ingestion pipeline
        pipeline.process_documents([document])

        logger.info("Text ingestion completed")

        return {
            "status": "success",
            "message": "Text successfully ingested into vector store"
        }

    except Exception as e:

        logger.error(f"Text ingestion failed: {e}")

        raise HTTPException(status_code=500, detail="Text ingestion failed")


@router.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the Corpora directory.

    The ingestion pipeline will be triggered by
    filesystem events (watcher/webhook).
    """

    try:

        saved_path = file_service.save_file(file)

        logger.info(f"Received file: {file.filename}")

        return {"status": "success", "message": "File uploaded successfully", "path": saved_path}

    except ValueError as e:

        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:

        logger.error(f"File upload failed: {e}")

        raise HTTPException(status_code=500, detail="File upload failed")


@router.post("/web")
async def ingest_web(request: WebIngestionRequest):

    ...
    