from pypdf import PdfReader

from src.rag.core.loaders.base_loader import BaseLoader


class PdfLoader(BaseLoader):
    extensions = [".pdf"]

    def extract_text(self, path):
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return " ".join(pages)
