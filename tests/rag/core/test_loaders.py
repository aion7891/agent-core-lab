import pytest
from docx import Document

from src.rag.core.loaders.loader_registry import load_document


def test_docx_returns_a_document_dict(tmp_path):
    path = tmp_path / "notes.docx"
    document = Document()
    document.add_paragraph("Hello remote work")
    document.add_paragraph("")
    document.add_paragraph("Second line")
    document.save(path)

    result = load_document(path)

    assert result == {
        "source": "notes.docx",
        "file_type": ".docx",
        "text": "Hello remote work Second line",
    }


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "notes.xyz"
    path.write_text("hi")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.pdf")
