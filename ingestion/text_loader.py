import re


class TextLoader:

    def load_text(self, text: str) -> str:
        """
        Main entry point for text cleaning pipeline
        """

        cleaned = self.clean_text(text)

        normalized = self.normalize_text(cleaned)

        return normalized


    def clean_text(self, text: str) -> str:
        """
        Remove extra whitespace and tabs
        """

        text = re.sub(r"\s+", " ", text)

        return text.strip()


    def normalize_text(self, text: str) -> str:
        """
        Normalize new lines and spacing
        """

        return text.replace("\n", " ")

