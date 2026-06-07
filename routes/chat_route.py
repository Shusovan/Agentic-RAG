from fastapi import APIRouter

from schemas.rag_schema import QueryRequest
from graph.workflow import Flow


router = APIRouter(prefix="/v1", tags=["Chat"])

graph = Flow()


@router.post("/query")
async def chat(request: QueryRequest):

    state = graph.run(request.query)

    return state.model_dump()