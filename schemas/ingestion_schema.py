from pydantic import BaseModel


class TextIngestionRequest(BaseModel):
    text: str