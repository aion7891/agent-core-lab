from docx import Document

from src.rag.core.loaders.base_loader import BaseLoader


class DocxLoader(BaseLoader):
    extensions = [".docx"]

    def extract_text(self, path):
        document = Document(path)
        paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
        return " ".join(paragraphs)
