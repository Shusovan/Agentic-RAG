from pathlib import Path

from ingestion.loader_registry import LoadRegistry


class DocumentLoader:

    def __init__(self):

        self.registry = LoadRegistry()

    def load_directory(self, directory: Path):

        documents = []

        for file in directory.iterdir():

            if not file.is_file():
                continue

            loader = self.registry.get_file_loader(file)

            if loader:
                documents.extend(loader.load(file))

        return documents

    def load_file(self, path: Path):

        loader = self.registry.get_file_loader(path)

        if loader is None:
            raise ValueError(f"No loader registered for {path.suffix}")

        return loader.load(path)

    def load_url(self, url: str):

        return self.registry.get_web_loader().load(url)