"""PDF processing module using pypdf.

Extracts text, preserves exact page numbering, splits into manageable
search chunks, and detects scanned/empty/corrupted/password-protected files.
"""

import os
import re
from typing import Any, Dict, List, Tuple
from pypdf import PdfReader
from pypdf.errors import EmptyFileError, FileNotDecryptedError, PdfReadError


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def chunk_page_text(text: str, max_chars: int = 600, overlap: int = 100) -> List[str]:
    """Splits single-page text into overlapping segments to preserve context."""
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        if end < len(text):
            last_punct = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("! "), chunk.rfind("; "))
            if last_punct > max_chars // 2:
                end = start + last_punct + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap
        if start >= len(text) - overlap:
            break
    return chunks


def process_pdf(file_path: str, filename: str) -> Tuple[str, int, List[Dict[str, Any]]]:
    """Reads and parses a PDF.

    Returns:
        (status_message, total_pages, chunks_list)
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return "Empty file: File size is 0 bytes.", 0, []

    try:
        reader = PdfReader(file_path)
    except EmptyFileError:
        return "Empty or corrupt PDF file.", 0, []
    except PdfReadError as e:
        return f"Cannot read PDF: {str(e)}", 0, []
    except Exception as e:
        return f"Unexpected read error: {str(e)}", 0, []

    if reader.is_encrypted:
        try:
            decrypt_status = reader.decrypt("")
            if decrypt_status == 0:
                return "Encrypted PDF: Password is required to open this file.", len(reader.pages), []
        except Exception:
            return "Encrypted PDF: Access restricted.", len(reader.pages), []

    total_pages = len(reader.pages)
    if total_pages == 0:
        return "PDF has 0 pages.", 0, []

    extracted_chunks: List[Dict[str, Any]] = []
    total_characters_found = 0

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        cleaned = clean_text(raw_text)
        total_characters_found += len(cleaned)

        page_chunks = chunk_page_text(cleaned)
        for chunk_idx, chunk_text in enumerate(page_chunks):
            extracted_chunks.append({
                "filename": filename,
                "page_number": page_num,
                "chunk_index": chunk_idx,
                "content": chunk_text
            })

    if total_characters_found < 30:
        return (
            "No extractable text found. This document appears to be a scanned image or empty. "
            "It may require an optical character recognition (OCR) pre-processing step.",
            total_pages,
            []
        )

    return "Ready", total_pages, extracted_chunks
