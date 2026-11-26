
"""PDF processing module using pypdf.

Preserves document structure, paragraph boundaries, likely headings,
exact page numbering, and manageable search chunks.
"""

import os
import re
from typing import Any, Dict, List, Tuple

from pypdf import PdfReader
from pypdf.errors import EmptyFileError, FileNotDecryptedError, PdfReadError


# ---------------------------------------------------------
# 1. NORMALIZE TEXT WITHOUT DESTROYING PARAGRAPH STRUCTURE
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize extracted PDF text while preserving paragraph breaks."""

    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")

    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \u00a0]+", " ", line).strip()
        lines.append(line)

    # Preserve paragraph boundaries but remove excessive blank lines.
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------
# 2. DETECT LIKELY HEADINGS
# ---------------------------------------------------------

def is_heading(line: str) -> bool:
    """Identify likely section headings using text patterns."""

    line = line.strip()

    if not line or len(line) > 120:
        return False

    # Numbered headings: 1. INTRODUCTION, 1.1 Background, etc.
    numbered_heading = re.match(
        r"^(?:CHAPTER\s+)?\d+(?:\.\d+)*\.?\s+.{2,}$",
        line,
        re.IGNORECASE
    )

    if numbered_heading:
        return True

    # Chapter headings.
    if re.match(r"^(CHAPTER|SECTION)\s+[IVXLCD0-9]+", line, re.IGNORECASE):
        return True

    # All-uppercase headings.
    letters = re.sub(r"[^A-Za-z]", "", line)

    if letters and letters.isupper() and len(line.split()) <= 12:
        return True

    # Short title-like lines without sentence-ending punctuation.
    if (
        len(line.split()) <= 7
        and len(line) <= 65
        and not re.search(r"[.!?;]$", line)
    ):
        return True

    return False


# ---------------------------------------------------------
# 3. CONVERT PAGE TEXT INTO PARAGRAPH BLOCKS
# ---------------------------------------------------------

def extract_paragraphs(text: str) -> List[str]:
    """
    Extract separate paragraphs and headings.

    Headings are kept separate so retrieval.py can later
    associate them with their following paragraphs.
    """

    text = clean_text(text)

    if not text:
        return []

    blocks = []
    current_paragraph = []

    def flush_paragraph():
        if current_paragraph:
            paragraph = " ".join(current_paragraph).strip()

            if paragraph:
                blocks.append(paragraph)

            current_paragraph.clear()

    for line in text.split("\n"):

        line = line.strip()

        # Blank line indicates a paragraph boundary.
        if not line:
            flush_paragraph()
            continue

        # A heading starts a new block.
        if is_heading(line):
            flush_paragraph()
            blocks.append(line)
        else:
            # Merge PDF line wrapping into readable paragraphs.
            current_paragraph.append(line)

    flush_paragraph()

    return blocks


# ---------------------------------------------------------
# 4. SPLIT ONLY VERY LONG PARAGRAPHS
# ---------------------------------------------------------

def split_long_paragraph(
    paragraph: str,
    max_chars: int = 600
) -> List[str]:
    """Split long paragraphs without unnecessarily breaking sentences."""

    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)

    chunks = []
    current = ""

    for sentence in sentences:

        if not sentence:
            continue

        # If a single sentence is too long, split at word boundaries.
        if len(sentence) > max_chars:

            if current:
                chunks.append(current.strip())
                current = ""

            words = sentence.split()
            piece = ""

            for word in words:
                candidate = f"{piece} {word}".strip()

                if len(candidate) > max_chars and piece:
                    chunks.append(piece.strip())
                    piece = word
                else:
                    piece = candidate

            if piece:
                current = piece

            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current.strip())

    return chunks


# ---------------------------------------------------------
# 5. CREATE SEARCH CHUNKS WHILE PRESERVING PARAGRAPHS
# ---------------------------------------------------------

def chunk_page_text(
    text: str,
    max_chars: int = 600,
    overlap: int = 100
) -> List[str]:
    """
    Create chunks from individual paragraphs and headings.

    Normal paragraphs remain intact.
    Only exceptionally long paragraphs are divided.
    """

    blocks = extract_paragraphs(text)

    chunks = []

    for block in blocks:
        block_chunks = split_long_paragraph(block, max_chars)

        for chunk in block_chunks:
            if chunk.strip():
                chunks.append(chunk.strip())

    return chunks


# ---------------------------------------------------------
# 6. PROCESS PDF
# ---------------------------------------------------------

def process_pdf(
    file_path: str,
    filename: str
) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Reads and parses a PDF.

    Returns:
        (status_message, total_pages, chunks_list)

    Each chunk retains:
        filename
        page_number
        chunk_index
        content
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

    # Handle password-protected PDFs.
    if reader.is_encrypted:
        try:
            decrypt_status = reader.decrypt("")

            if decrypt_status == 0:
                return (
                    "Encrypted PDF: Password is required to open this file.",
                    len(reader.pages),
                    []
                )

        except Exception:
            return "Encrypted PDF: Access restricted.", 0, []

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

        # Preserve original paragraph and heading structure.
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

    # Detect scanned or empty PDFs.
    if total_characters_found < 30:

        return (
            "No extractable text found. This document appears to be "
            "a scanned image or empty. It may require OCR processing.",
            total_pages,
            []
        )

    return "Ready", total_pages, extracted_chunks
