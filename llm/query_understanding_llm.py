import json
import logging
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from prompts.query_understanding_prompt import QUERY_UNDERSTANDING_PROMPT


logger = logging.getLogger(__name__)

load_dotenv()


class QueryUnderstandingLLM:
    """
        Focused LLM wrapper — json_object mode only.
        Extracts structured metadata from the raw user query.
    """

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b"):
        
        self.client = Groq(api_key=api_key)
        self.model = model


    def analyze_query(self, query: str) -> dict[str, Any]:

        logger.info("[QueryUnderstandingLLM] Calling Groq API")

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": QUERY_UNDERSTANDING_PROMPT}, {"role": "user", "content": f"Query: {query}"}],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        raw = completion.choices[0].message.content.strip()

        logger.debug(f"[QueryUnderstandingLLM] Raw LLM response: {raw}")

        return self._parse_and_validate(raw)
    

    def _parse_and_validate(self, content: str) -> dict[str, Any]:

        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)

            if not match:
                raise ValueError("No JSON object found in LLM response")
            
            parsed = json.loads(match.group(0))

        except Exception as e:
            logger.error(f"[QueryUnderstandingLLM] Failed to parse JSON: {content}")
            raise ValueError("Failed to parse LLM JSON output")
        
        required = ["intent", "topic", "entities", "source", "rewritten_query"]

        missing_fields = [field for field in required if field not in parsed]

        if missing_fields:
            logger.error(f"[QueryUnderstandingLLM] Missing fields in LLM response: {missing_fields}")
            raise ValueError(f"LLM response missing required fields: {missing_fields}")
        
        return parsed