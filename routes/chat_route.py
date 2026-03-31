from fastapi import APIRouter

from schemas.query_schema import QueryRequest
from graph.workflow import Flow


router = APIRouter(prefix="/v1", tags=["Chat"])

graph = Flow()


@router.post("/query")
async def chat(request: QueryRequest):

    state = graph.run(request.query)

    return {"structured_query": state.structured_query}