"""Extract clean text from a JD or CV file.

PDF via pdfplumber; .txt/.md read directly. Empty extraction is surfaced as a
clear error (scanned/image PDFs are out of scope — no OCR).
"""
from __future__ import annotations

import os

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class ExtractionError(Exception):
    """Raised when a file cannot be read or yields no usable text."""


def extract_text(path: str) -> str:
    """Return cleaned text for a supported file, or raise ExtractionError."""
    if not os.path.isfile(path):
        raise ExtractionError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ExtractionError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    text = _extract_pdf(path) if ext == ".pdf" else _read_text_file(path)
    text = text.strip()
    if not text:
        raise ExtractionError(
            f"No text extracted from '{os.path.basename(path)}'. "
            "If this is a scanned/image PDF, OCR is out of scope for this tool."
        )
    return text


def _extract_pdf(path: str) -> str:
    import pdfplumber  # imported lazily so .txt-only flows don't need the dep

    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                parts.append(page_text)
    return "\n\n".join(parts)


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()
