import shutil
import logging
from pathlib import Path
import uuid
from fastapi import UploadFile

from config.settings import settings


logger = logging.getLogger(__name__)


class FileService:

    # Maximum file upload
    MAX_FILES = 20

    # 100 MB total for normal multi-file uploads
    MAX_TOTAL_UPLOAD_SIZE = 100 * 1024 * 1024

    # Maximum large file = 10 GB
    MAX_LARGE_FILE_SIZE = 10 * 1024 * 1024 * 1024

    # Each chunk = 100 MB
    CHUNK_SIZE = 100 * 1024 * 1024

    # Directories
    CORPORA_DIR = Path("Corpora")
    LARGE_UPLOAD_TEMP_DIR = Path("upload_temp")

    def __init__(self):

        self.CORPORA_DIR.mkdir(parents=True, exist_ok=True)
        self.LARGE_UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)


    # save single file
    def save_file(self, file: UploadFile) -> str:
        """
        Save uploaded file to Corpora directory
        """

        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_extension}")

        file_path = self.CORPORA_DIR / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File saved to {file_path}")

        return str(file_path)


    # save multiple files
    async def save_multiple_files(self, files: list[UploadFile]):
        """
            Save multiple normal-sized files.

            Maximum:
                20 files
                100 MB total
        """

        if not files:
            raise ValueError("At least one file must be uploafed")

        if len(files) > self.MAX_FILES:
            raise ValueError(f"Maximum {self.MAX_FILES} files can be uploaded at once")

        # calculate total upload size
        total_size = 0

        for file in files:
            try:
                file.file.seek(0,2)
                file_size = file.file.tell()
                file.file.seek(0)
                total_size += file_size

            except Exception as exc:
                logger.error("Unable to determine size of %s: %s", file.filename, exc)
                raise ValueError(f"Unable to determine file size: {file.filename}")

        if total_size > self.MAX_TOTAL_UPLOAD_SIZE:

            raise ValueError("Total upload size exceeds "
                             f"{self.MAX_TOTAL_UPLOAD_SIZE / (1024 * 1024):.0f} MB")

        # Save individual files
        uploaded_files = []
        failed_files = []

        for file in files:

            try:
                saved_path = self.save_file(file)
                uploaded_files.append({"filename": file.filename, "path": saved_path})

            except ValueError as exc:
                logger.error("Failed to upload %s: %s",file.filename, exc)
                failed_files.append({"filename": file.filename, "error": str(exc)})

            except Exception:
                logger.exception("Upload failed: %s", file.filename)
                failed_files.append({"filename": file.filename, "error": "File upload failed"})

        return {
            "total_files": len(files),
            "uploaded_count": len(uploaded_files),
            "failed_count": len(failed_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "uploaded_files": uploaded_files,
            "failed_files": failed_files
        }


    # upload large files
    def start_large_upload(self, filename: str, file_size: int):
        """
            Create large file upload session
        """

        if not filename:
            raise ValueError("Filename is required")

        if file_size <= 0:
            raise ValueError("file size must be more than 0")

        if file_size > self.MAX_LARGE_FILE_SIZE:
            raise ValueError("Maximum upload limit is 10GB")


        # sanitize file
        safe_filename = Path(filename).name

        # generate upload_id
        upload_id = str(uuid.uuid4())

        upload_dir = (self.LARGE_UPLOAD_TEMP_DIR / upload_id)

        upload_dir.mkdir(parents=True, exist_ok=True)

        # store metadata
        metadata_file = (upload_dir / "metadata.txt")

        metadata_file.write_text(f"filename={safe_filename}\n"
            f"file_size={file_size}\n"
            f"chunk_size={self.CHUNK_SIZE}\n")

        logger.info("Large upload started: %s | %s | %s bytes",
            upload_id,
            safe_filename,
            file_size)

        return {
            "upload_id": upload_id,
            "filename": safe_filename,
            "file_size": file_size,
            "chunk_size": self.CHUNK_SIZE,
            "max_file_size": self.MAX_LARGE_FILE_SIZE
        }


    # uoloud large diles by breaking them into chunks
    async def upload_large_file_chunk(self, upload_id: str, chunk_number: int, file: UploadFile):
        """
        Save one chunk of a large file.
        """

        if chunk_number < 0:
            raise ValueError("Invalid chunk number")

        upload_dir = (self.LARGE_UPLOAD_TEMP_DIR / upload_id)

        if not upload_dir.exists():
            raise FileNotFoundError("Upload session not found")

        chunk_path = (upload_dir / f"chunk_{chunk_number}")

        # Don't allow duplicate chunk uploads
        if chunk_path.exists():
            raise FileExistsError(f"Chunk {chunk_number} ""already exists")

        bytes_written = 0

        try:
            with chunk_path.open("wb") as buffer:

                while True:
                    data = await file.read(1024 * 1024)

                    if not data:
                        break

                    bytes_written += len(data)

                    # Prevent chunk from exceeding 100 MB
                    if bytes_written > self.CHUNK_SIZE:

                        buffer.close()
                        chunk_path.unlink(missing_ok=True)

                        raise ValueError("Chunk exceeds maximum "
                            f"size of "
                            f"{self.CHUNK_SIZE / (1024 * 1024):.0f} MB")

                    buffer.write(data)

            logger.info("Chunk uploaded | upload=%s | chunk=%s | size=%s",
                upload_id, chunk_number, bytes_written)

            return {
                "upload_id": upload_id,
                "chunk_number": chunk_number,
                "chunk_size": bytes_written
            }

        except Exception:
            chunk_path.unlink(missing_ok=True)
            raise



    def complete_large_upload(self, upload_id: str):
        """
        Combine all chunks into the final file
        and move it into Corpora.
        """

        upload_dir = (self.LARGE_UPLOAD_TEMP_DIR / upload_id)

        if not upload_dir.exists():
            raise FileNotFoundError("Upload session not found")

        metadata_file = (upload_dir / "metadata.txt")

        if not metadata_file.exists():
            raise ValueError("Upload metadata not found")


        # Read metadata
        metadata = {}

        for line in metadata_file.read_text().splitlines():
            key, value = line.split("=", 1)
            metadata[key] = value

        filename = metadata["filename"]

        expected_size = int(metadata["file_size"])


        # Find chunks
        chunk_files = list(upload_dir.glob("chunk_*"))

        if not chunk_files:
            raise ValueError("No chunks have been uploaded")
        
        # Sort chunks numerically
        chunk_files.sort(key=lambda path: int(path.name.split("_")[1]))


        # Validate chunk sequence
        for expected_number, chunk_path in enumerate(chunk_files):
            actual_number = int(chunk_path.name.split("_")[1])

            if actual_number != expected_number:
                raise ValueError(f"Missing chunk. "f"Expected chunk "f"{expected_number}")


        # Assemble file
        temp_final_path = (upload_dir / f"completed_{filename}")

        total_size = 0

        try:
            with temp_final_path.open("wb") as final_file:
                for chunk_path in chunk_files:
                    with chunk_path.open("rb") as chunk_file:
                        while True:
                            data = chunk_file.read(1024 * 1024)

                            if not data:
                                break

                            final_file.write(data)
                            total_size += len(data)

            # Verify final size
            if total_size != expected_size:
                temp_final_path.unlink(missing_ok=True)

                raise ValueError(
                    f"File size mismatch. "
                    f"Expected {expected_size} bytes, "
                    f"received {total_size} bytes")


            # Final destination
            destination = (self.CORPORA_DIR / filename)

            # Prevent overwrite
            if destination.exists():

                stem = destination.stem
                suffix = destination.suffix

                destination = (self.CORPORA_DIR / f"{stem}_{uuid.uuid4().hex[:8]}" f"{suffix}")


            # Move completed file
            shutil.move(str(temp_final_path), str(destination))


            # Cleanup upload session
            shutil.rmtree(upload_dir, ignore_errors=True)

            logger.info("Large file completed: %s", destination)

            return {
                    "upload_id": upload_id,
                    "filename": destination.name,
                    "path": str(destination),
                    "size_bytes": total_size,
                    "size_gb": round(total_size / (1024 ** 3), 2)
                }

        except Exception:
            # Keep chunks for possible debugging/retry.
            # Do not delete the entire upload session here.
            logger.exception("Failed to complete large upload: %s", upload_id)
            raise


# Service instance
file_service = FileService()