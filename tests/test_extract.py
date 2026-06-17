"""Extraction: plain-text read + clear errors (PDF path needs pdfplumber)."""
import pytest

from src.extract import ExtractionError, extract_text


def test_reads_txt(tmp_path):
    p = tmp_path / "cv.txt"
    p.write_text("Senior Backend Engineer\nBuilt payment services.", encoding="utf-8")
    assert "payment services" in extract_text(str(p))


def test_reads_md(tmp_path):
    p = tmp_path / "jd.md"
    p.write_text("# Role\nWe need a backend engineer.", encoding="utf-8")
    assert "backend engineer" in extract_text(str(p))


def test_missing_file_raises():
    with pytest.raises(ExtractionError):
        extract_text("does_not_exist.txt")


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "cv.docx"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ExtractionError):
        extract_text(str(p))


def test_empty_extraction_is_surfaced(tmp_path):
    p = tmp_path / "blank.txt"
    p.write_text("   \n\t  ", encoding="utf-8")
    with pytest.raises(ExtractionError):
        extract_text(str(p))
