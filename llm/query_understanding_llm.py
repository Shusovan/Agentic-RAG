import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from sympy import content


logger = logging.getLogger(__name__)

load_dotenv()


class QueryUnderstandingLLM:

    def __init__(self, api_key: str, model: str = "qwen/qwen3-32b"):

        # api_key = os.getenv("GROQ_API_KEY")

        # if not api_key:
            #raise ValueError("GROQ_API_KEY not set in environment")
        
        self.client = Groq(api_key=api_key)
        self.model = model


    def analyze_query(self, query: str) -> dict[str, Any]:
        """
        Sends prompt to Groq LLM and returns structured JSON response
        """

        try:
            logger.info(f"Analyzing query with LLM")

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Query Understanding Agent in an enterprise RAG system. Analyze the user query and extract intent, topic, entities, source, and rewritten_query. Return ONLY valid JSON with no explanation or markdown."
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}"
                    }
                ],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"}
            )
            
            content = completion.choices[0].message.content.strip()

            logger.debug(f"LLM response: {content}")

            parsed = self._parse_json(content)
            self._validate_response(parsed)

            return parsed
        
        except Exception as e:
            logger.error(f"Groq LLM generation failed: {e}")
            raise


    
    def _parse_json(self, content: str) -> dict[str, Any]:
        """
        Parse the LLM response as JSON, handling common formatting issues
        """
        try:
            # Extract first JSON object
            match = re.search(r"\{.*\}", content, re.DOTALL)
            
            if not match:
                raise ValueError("No JSON object found")

            response = match.group(0)

            return json.loads(response)

        except Exception:
            logger.error(f"Invalid JSON from LLM: {content}")
            raise ValueError("Failed to parse LLM JSON output")
        
        '''try:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            if content.startswith("```"):
                content = content.strip("```")
                content = content.replace("json", "", 1).strip()

            return json.loads(content)


        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from LLM: {content}")
            raise ValueError("Failed to parse LLM JSON output")'''
        
    
    def _validate_response(self, response: dict[str, Any]):
        """
        Validate required schema fields
        """
        required_keys = [
            "rewritten_query",
            "intent",
            "topic",
            "entities",
            "source"
        ]

        missing = [k for k in required_keys if k not in response]

        if missing:
            raise ValueError(f"Missing keys in LLM response: {missing}")