"""
Picks the right loader for a file based on its extension.
Add new loaders to the LOADERS list.
"""

import os

from src.rag.core.loaders.docx_loader import DocxLoader
from src.rag.core.loaders.pdf_loader import PdfLoader

LOADERS = [
    PdfLoader(),
    DocxLoader(),
]


def get_loader(path):
    extension = os.path.splitext(str(path))[1].lower()

    for loader in LOADERS:
        if extension in loader.extensions:
            return loader

    supported = [ext for loader in LOADERS for ext in loader.extensions]
    raise ValueError(f"Unsupported file type '{extension}'. Supported: {supported}")


def load_document(path):
    return get_loader(path).load(path)
