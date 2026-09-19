import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from langchain_core.documents import Document

from slowapi import Limiter
from slowapi.util import get_remote_address

from services.ingestion_service import file_service
from ingestion import ingestion_pipeline
from schemas.ingestion_schema import TextIngestionRequest, WebIngestionRequest

from config.vector_dependency import vector_store, embedding_pipeline


router = APIRouter()

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


pipeline = ingestion_pipeline.IngestionPipeline(vector_store=vector_store, embedding_pipeline=embedding_pipeline)


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


@router.post("/upload-documents")
@limiter.limit("10/minute")
async def upload_documents(request: Request, files: list[UploadFile] = File(...)):
    """
    Upload multiple normal-sized documents.

    Limits:
        - Maximum 20 files
        - Maximum 100 MB total
        - Maximum 10 requests/minute/IP
    """

    try:
        result = await file_service.save_multiple_files(files)

        if result["uploaded_count"] == 0:
            raise HTTPException(status_code=500,
                detail={
                    "message": ("No files were uploaded successfully"),
                    "failed_files": (result["failed_files"])
                }
            )

        if result["failed_count"] > 0:
            result["status"] = ("partial_success")
            result["message"] = (f"{result['uploaded_count']} file(s) uploaded successfully, "
                f"{result['failed_count']} file(s) failed")

        else:
            result["status"] = "success"
            result["message"] = (f"{result['uploaded_count']} file(s) uploaded successfully")

        return result

    except HTTPException:
        raise

    except ValueError as exc:

        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        logger.exception("Multiple document upload failed")
        raise HTTPException(status_code=500, detail="File upload failed")


# LARGE FILE - START UPLOAD
@router.post("/large-upload/start")
@limiter.limit("5/minute")
async def start_large_upload(request: Request, filename: str, file_size: int):
    """
    Start a resumable large-file upload.

    Maximum file size:
        10 GB

    Recommended chunk size:
        100 MB
    """

    try:
        result = file_service.start_large_upload(filename=filename, file_size=file_size)

        return {
            "status": "success",
            "message": "Large upload started",
            **result
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        logger.exception("Failed to start large upload")
        raise HTTPException(status_code=500, detail="Failed to start large upload")


# LARGE FILE - UPLOAD CHUNK
@router.put("/large-upload/{upload_id}/chunk/{chunk_number}")
async def upload_large_file_chunk(upload_id: str, chunk_number: int, file: UploadFile = File(...)):
    """
    Upload one chunk of a large file.

    Chunk size:
        100 MB
    """

    try:
        result = await file_service.upload_large_file_chunk(
            upload_id=upload_id,
            chunk_number=chunk_number,
            file=file)

        return {
            "status": "success",
            "message": ("Chunk uploaded successfully"),
            **result
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        logger.exception("Chunk upload failed | upload=%s | chunk=%s", upload_id, chunk_number)
        raise HTTPException(status_code=500, detail="Chunk upload failed")


# LARGE FILE - COMPLETE
@router.post("/large-upload/{upload_id}/complete")
async def complete_large_upload(upload_id: str):
    """
    Complete a large-file upload.

    All chunks are combined and the final file
    is moved into the Corpora directory.
    """

    try:
        result = file_service.complete_large_upload(upload_id)

        return {
            "status": "success",
            "message": ("Large file uploaded successfully"),
            **result
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        logger.exception("Failed to complete large upload: %s", upload_id)
        raise HTTPException(status_code=500, detail="Failed to complete large upload")


@router.post("/web")
async def ingest_web(request: WebIngestionRequest):

    ...
    