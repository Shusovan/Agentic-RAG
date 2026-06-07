from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums — single source of truth for intent/source/route ───────────────────

class Intent(str, Enum):
    INTERNAL_QUESTION = "INTERNAL_QUESTION"
    DOCUMENT_QUERY    = "DOCUMENT_QUERY"
    GENERAL_QUESTION  = "GENERAL_QUESTION"
    GREETING          = "GREETING"
    CHITCHAT          = "CHITCHAT"
    AMBIGUOUS         = "AMBIGUOUS"
    OUT_OF_SCOPE      = "OUT_OF_SCOPE"
    UNKNOWN           = "UNKNOWN"


class Source(str, Enum):
    INTERNAL = "INTERNAL"
    GENERAL  = "GENERAL"
    UNKNOWN  = "UNKNOWN"


class Route(str, Enum):
    RAG      = "rag"
    LLM      = "llm"
    CLARIFY  = "clarify"
    FALLBACK = "fallback"
    ERROR    = "error"


# ── Request ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Raw user query")

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


# ── Step 1 output: QueryUnderstandingAgent ────────────────────────────────────

class StructuredQuery(BaseModel):
    original_query: str  = Field(..., description="Raw user query as received")
    revised_query:  str  = Field(..., description="Rewritten query optimised for vector search")
    intent:         Intent = Field(..., description="Classified intent of the query")
    topic:          str  = Field(..., description="Main subject of the query")
    entities:       list[str] = Field(default_factory=list, description="Key entities extracted")
    source:         Source = Field(..., description="Inferred knowledge source")

    @field_validator("intent", mode="before")
    @classmethod
    def normalise_intent(cls, v: str) -> str:
        """Uppercase and strip before enum validation."""
        return v.upper().strip() if isinstance(v, str) else v

    @field_validator("source", mode="before")
    @classmethod
    def normalise_source(cls, v: str) -> str:
        return v.upper().strip() if isinstance(v, str) else v

    @field_validator("entities", mode="before")
    @classmethod
    def coerce_entities(cls, v: Any) -> list[str]:
        """Always produce a clean list — handles LLM returning a string."""
        if isinstance(v, list):
            return [str(e).strip() for e in v if str(e).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    @model_validator(mode="after")
    def entities_never_empty(self) -> StructuredQuery:
        """Fall back to topic if LLM returned no entities."""
        if not self.entities and self.topic:
            self.entities = [self.topic]
        return self


# ── Retrieved document ────────────────────────────────────────────────────────

class RetrievedDocument(BaseModel):
    id:               str
    content:          str
    metadata:         dict[str, Any] = Field(default_factory=dict)
    similarity_score: float = Field(ge=0.0, le=1.0)
    rank:             int   = Field(ge=1)


# ── Final pipeline output ─────────────────────────────────────────────────────

class RAGResponse(BaseModel):
    original_query:      str
    revised_query:       str
    intent:              Intent
    topic:               str
    entities:            list[str]
    source:              Source
    route:               Route
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    generated_answer:    str | None = None
    error:               str | None = None