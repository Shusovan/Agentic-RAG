from typing import Any

import logging

from prompts.query_understanding_prompt import QUERY_UNDERSTANDING_PROMPT
from llm.query_understanding_llm import QueryUnderstandingLLM


logger = logging.getLogger(__name__)


class QueryUnderstandingAgent:

    def __init__(self, llm: QueryUnderstandingLLM):
        self.llm = llm

    
    def process_query(self, query: str) -> dict[str, Any]:
        """
        Main method to process user query and return structured understanding
        """

        try:
            logger.info(f"Processing query: {query}")

            prompt = QUERY_UNDERSTANDING_PROMPT.format(query=query)

            for attempt in range(3):
                try:
                    response = self.llm.analyze_query(query)
                    self._validate_response(response)
                    break

                except Exception as e:
                    logger.warning(f"Retry {attempt+1}: {e}")

            else:
                raise RuntimeError("LLM failed after retries")
            

            # Normalize intent and source to uppercase for consistency
            response["intent"] = (response.get("intent") or "UNKNOWN").upper()
            response["source"] = (response.get("source") or "UNKNOWN").upper()

            if not isinstance(response.get("entities"), list):
                response["entities"] = [response["entities"]] if response.get("entities") else []

            
            structured_query = {
                "original_query": query,
                "revised_query": response["rewritten_query"],
                "intent": response["intent"],
                "topic": response["topic"],
                "entities": response["entities"],
                "source": response["source"]
            }

            logger.info(f"Query understanding successful: {structured_query}")

            return structured_query

        except Exception as e:
            logger.error(f"Query understanding failed: {e}")
            raise

    
    def _validate_response(self, response: dict[str, Any]):

        required_keys = [
            "rewritten_query",
            "intent",
            "topic",
            "entities",
            "source"
        ]

        missing = [k for k in required_keys if k not in response]

        if missing:
            raise ValueError(f"Missing keys: {missing}")

        source = response.get("source")
        
        '''if not source or source.upper():
            response["source"] = "AUTO"
        
        else:
            response["source"] = source.upper()'''

        # ENTITIES (never empty)
        entities = response.get("entities")

        if not entities or not isinstance(entities, list):
            response["entities"] = [response["topic"]]

        # Clean entities
        response["entities"] = [e.strip() for e in response["entities"] if e.strip()]