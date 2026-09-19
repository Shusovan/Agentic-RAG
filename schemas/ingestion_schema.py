from pydantic import BaseModel, HttpUrl


class TextIngestionRequest(BaseModel):
    text: str

class WebIngestionRequest(BaseModel):
    url: HttpUrl