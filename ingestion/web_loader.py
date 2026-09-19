class WebLoader:

    def __init__(self, url: str):
        self.url = url

    def load(self) -> str:
        import requests

        response = requests.get(self.url)
        response.raise_for_status()
        return response.text