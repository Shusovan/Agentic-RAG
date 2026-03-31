from pydantic import BaseModel
from typing import List


class StructuredQuery(BaseModel):

    original_query : str

    revised_query : str

    intent : str

    topic : str

    entities : List[str]

    source : str


class QueryRequest(BaseModel):

    query : str