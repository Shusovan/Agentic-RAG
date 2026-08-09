from groq import Groq
import logging


logger = logging.getLogger(__name__)


class GenerativeLLM:
    """
    Direct LLM generation — used for GENERAL / GREETING / CHITCHAT routes.
    No retrieval, no tools. Pure conversational response.
    """

    SYSTEM_PROMPT = """You are a helpful enterprise assistant. 
                       Answer the user's question clearly and concisely.
                       If it is a greeting or casual message, respond naturally and warmly."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model


    def generate_response(self, query: str, context: str | None = None) -> str:

        logger.info("[GenerativeLLM] Calling Groq API for response generation")

        user_content = query

        if context:
            user_content = f"Context: {context}\n\nQuestion: {query}"

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        response = completion.choices[0].message.content.strip()

        logger.debug(f"[GenerativeLLM] Response generated Successfully: {response}")

        return response