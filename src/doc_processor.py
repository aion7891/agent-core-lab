import os
import csv
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import ollama
import numpy as np


# =====================================================================
# STEP 1 UPGRADE: MULTI-FORMAT PARSER & NORMALIZER
# =====================================================================

class MultiFormatParser:
    def parse(self, path) -> dict:
        ext = os.path.splitext(path)[1].lower()
        file_name = os.path.basename(path)
        raw_data = ''

        output_dict = {
            "source" : file_name,
            "file_type" : ext,
            "text" : raw_data
        }

        if ext == ".pdf":
            raw_data = self._parse_pdf(path)
            output_dict['text'] = raw_data
            return output_dict

        if ext == ".docx":
            raw_data = self._parse_docx(path)
            output_dict['text'] = raw_data
            return raw_data

        raise Exception("Unsupported file type")

    def _parse_pdf(self, path) -> dict:
        reader = PdfReader(path)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return " ".join(pages_text)


    def _parse_docx(self, path) -> dict:
        doc = Document(path)
        pages_text = [para.text for para in doc.paragraph if para.text.strip()]
        return " ".join(pages_text)