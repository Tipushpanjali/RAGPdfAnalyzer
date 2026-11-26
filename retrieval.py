
"""Retrieval and Extractive Answering Engine.

Supports:
- Exact heading matching
- Similar heading matching
- Complete section extraction
- Paragraph-level BM25 fallback
- Verified filename and page citations
- Fully offline operation
"""

import math
import os
import pickle
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


# ---------------------------------------------------------
# 1. TOKENIZATION
# ---------------------------------------------------------

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def normalize_heading(text: str) -> str:
    """Normalize headings for comparison."""

    text = text.lower().strip()

    # Remove chapter and section numbering.
    text = re.sub(r"^(chapter|section)\s+[ivxlcd0-9]+\.?\s*", "", text)
    text = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", text)

    # Remove punctuation.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# 2. HEADING DETECTION
# ---------------------------------------------------------

def is_heading(text: str) -> bool:
    """Identify heading blocks created by pdf_processor.py."""

    text = text.strip()

    if not text or len(text) > 120:
        return False

    # Numbered headings.
    if re.match(
        r"^(?:(?:CHAPTER|SECTION)\s+)?\d+(?:\.\d+)*\.?\s+.{2,}$",
        text,
        re.IGNORECASE
    ):
        return True

    # Chapter headings.
    if re.match(
        r"^(CHAPTER|SECTION)\s+[IVXLCD0-9]+",
        text,
        re.IGNORECASE
    ):
        return True

    # All-uppercase headings.
    letters = re.sub(r"[^A-Za-z]", "", text)

    if letters and letters.isupper() and len(text.split()) <= 12:
        return True

    # Short title-like headings.
    if (
        len(text.split()) <= 7
        and len(text) <= 65
        and not re.search(r"[.!?;]$", text)
    ):
        return True

    return False


# ---------------------------------------------------------
# 3. HEADING SIMILARITY
# ---------------------------------------------------------

def heading_similarity(query: str, heading: str) -> float:
    """Compare a user query with a PDF heading."""

    q = normalize_heading(query)
    h = normalize_heading(heading)

    if not q or not h:
        return 0.0

    if q == h:
        return 1.0

    # Exact phrase contained within a longer heading.
    if q in h or h in q:
        return 0.95

    # Character-level similarity.
    similarity = SequenceMatcher(None, q, h).ratio()

    # Token overlap.
    q_tokens = set(tokenize(q))
    h_tokens = set(tokenize(h))

    if q_tokens:
        overlap = len(q_tokens.intersection(h_tokens)) / len(q_tokens)
        similarity = max(similarity, overlap * 0.9)

    return similarity


# ---------------------------------------------------------
# 4. SENTENCE SPLITTING
# ---------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    raw_sentences = re.split(r"(?<=[.?!;])\s+", text)

    clean_sents = [
        s.strip()
        for s in raw_sentences
        if len(s.strip()) > 10
    ]

    return clean_sents if clean_sents else [text.strip()]


# ---------------------------------------------------------
# 5. SECTION EXTRACTION
# ---------------------------------------------------------

def build_document_sections(
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Group paragraphs under their preceding heading.

    A section ends when the next heading appears.
    Sections can continue across PDF pages.
    """

    # Preserve actual document order.
    ordered_chunks = sorted(
        chunks,
        key=lambda c: (
            c.get("doc_id", 0),
            c.get("page_number", 0),
            c.get("chunk_index", 0)
        )
    )

    sections = []
    current_section = None

    for chunk in ordered_chunks:

        content = chunk.get("content", "").strip()

        if not content:
            continue

        if is_heading(content):

            # Save the previous section.
            if current_section and current_section["paragraphs"]:
                sections.append(current_section)

            current_section = {
                "doc_id": chunk.get("doc_id"),
                "filename": chunk["filename"],
                "heading": content,
                "start_page": chunk["page_number"],
                "paragraphs": [],
                "page_numbers": []
            }

        elif current_section is not None:

            # Do not mix content from different documents.
            if chunk.get("doc_id") != current_section["doc_id"]:
                if current_section["paragraphs"]:
                    sections.append(current_section)
                current_section = None
                continue

            current_section["paragraphs"].append(content)

            if chunk["page_number"] not in current_section["page_numbers"]:
                current_section["page_numbers"].append(chunk["page_number"])

    # Save the final section.
    if current_section and current_section["paragraphs"]:
        sections.append(current_section)

    return sections


# ---------------------------------------------------------
# 6. BM25 INDEX
# ---------------------------------------------------------

class UserBM25Index:

    def __init__(self, username: str, cache_dir: str = "indexes"):

        self.username = username
        self.cache_dir = cache_dir

        self.cache_file = os.path.join(
            cache_dir,
            f"{username}_bm25.pkl"
        )

        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict[str, Any]] = []

    def build_or_reload(
        self,
        all_user_chunks: List[Dict[str, Any]]
    ) -> None:

        os.makedirs(self.cache_dir, exist_ok=True)

        self.chunks = all_user_chunks

        if not self.chunks:

            self.bm25 = None

            if os.path.exists(self.cache_file):
                try:
                    os.remove(self.cache_file)
                except OSError:
                    pass

            return

        tokenized_corpus = [
            tokenize(c["content"])
            for c in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(
                    {
                        "chunks": self.chunks,
                        "bm25": self.bm25
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL
                )

        except Exception:
            pass

    def load_from_cache(self) -> bool:

        if os.path.exists(self.cache_file):

            try:
                with open(self.cache_file, "rb") as f:
                    data = pickle.load(f)

                    self.chunks = data.get("chunks", [])
                    self.bm25 = data.get("bm25")

                    return True

            except Exception:
                return False

        return False

    # -----------------------------------------------------
    # 7. SEARCH HEADINGS FIRST, THEN BM25
    # -----------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 4
    ) -> List[Tuple[Dict[str, Any], float]]:

        if not self.chunks:
            return []

        results = []

        # First: search for matching section headings.
        sections = build_document_sections(self.chunks)

        heading_matches = []

        for section in sections:

            similarity = heading_similarity(
                question,
                section["heading"]
            )

            if similarity >= 0.72:

                section_text = "\n\n".join(
                    section["paragraphs"]
                )

                if not section_text.strip():
                    continue

                section_result = {
                    "filename": section["filename"],
                    "page_number": section["start_page"],
                    "content": section_text,
                    "section_title": section["heading"],
                    "section_pages": section["page_numbers"],
                    "is_section": True
                }

                heading_matches.append(
                    (section_result, similarity)
                )

        if heading_matches:

            heading_matches.sort(
                key=lambda x: x[1],
                reverse=True
            )

            # Prefer heading matches over individual paragraphs.
            return heading_matches[:top_k]

        # Second: use BM25 if no suitable heading exists.
        if not self.bm25:
            return []

        tokens = tokenize(question)

        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        scored_pairs = [
            (self.chunks[i], float(score))
            for i, score in enumerate(scores)
        ]

        scored_pairs = [
            p for p in scored_pairs
            if p[1] > 0.0
        ]

        scored_pairs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return scored_pairs[:top_k]


# ---------------------------------------------------------
# 8. EXTRACT ANSWER
# ---------------------------------------------------------

def extract_answer(
    query_text: str,
    retrieved_passages: List[Tuple[Dict[str, Any], float]],
    score_threshold: float = 0.5
) -> Tuple[str, List[Dict[str, Any]]]:

    if not retrieved_passages:
        return (
            "I could not find a matching section or relevant passage "
            "in your uploaded documents.",
            []
        )

    # -----------------------------------------------------
    # FIRST PRIORITY: RETURN THE COMPLETE MATCHED SECTION
    # -----------------------------------------------------

    section_results = [
        (chunk, score)
        for chunk, score in retrieved_passages
        if chunk.get("is_section")
    ]

    if section_results:

        best_chunk, best_score = section_results[0]

        if best_score < 0.72:
            return (
                "I could not find a sufficiently similar section heading.",
                []
            )

        title = best_chunk["section_title"]
        content = best_chunk["content"]

        pages = best_chunk.get("section_pages", [])

        formatted_answer = (
            f"### {title}\n\n"
            f"{content}\n\n"
            f"**Source:** {best_chunk['filename']}\n\n"
            f"**Pages:** {', '.join(map(str, pages))}"
        )

        citations = []

        for page in pages:
            citations.append({
                "filename": best_chunk["filename"],
                "page_number": page,
                "passage": content
            })

        return formatted_answer, citations

    # -----------------------------------------------------
    # SECOND PRIORITY: BM25 PARAGRAPH SEARCH
    # -----------------------------------------------------

    top_score = retrieved_passages[0][1]

    if top_score < score_threshold:
        return (
            "I could not find a sufficiently relevant passage "
            "in your uploaded documents.",
            []
        )

    q_tokens = set(tokenize(query_text))

    extracted_entries = []
    seen_sentences = set()

    for chunk, chunk_score in retrieved_passages:

        sentences = split_sentences(chunk["content"])

        sentence_scores = []

        for sentence in sentences:

            s_tokens = set(tokenize(sentence))

            if not s_tokens:
                continue

            common = q_tokens.intersection(s_tokens)

            overlap_score = len(common) / (len(q_tokens) + 1e-6)

            if overlap_score > 0:
                sentence_scores.append(
                    (sentence, overlap_score)
                )

        sentence_scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        for sentence, score in sentence_scores[:2]:

            normalized = sentence.lower().strip()

            if (
                normalized not in seen_sentences
                and score >= 0.15
            ):

                seen_sentences.add(normalized)

                extracted_entries.append({
                    "sentence": sentence,
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "score": score
                })

    if not extracted_entries:

        best_chunk = retrieved_passages[0][0]

        fallback_text = (
            f"According to **{best_chunk['filename']}** "
            f"(Page {best_chunk['page_number']}):\n\n"
            f"> {best_chunk['content']}"
        )

        citation = [{
            "filename": best_chunk["filename"],
            "page_number": best_chunk["page_number"],
            "passage": best_chunk["content"]
        }]

        return fallback_text, citation

    lines = [
        "Based on your documents, here are the relevant findings:\n"
    ]

    citations = []

    for item in extracted_entries[:4]:

        lines.append(
            f"- \"{item['sentence']}\" — "
            f"*[{item['filename']}, Page {item['page_number']}]*"
        )

        citations.append({
            "filename": item["filename"],
            "page_number": item["page_number"],
            "passage": item["sentence"]
        })

    return "\n".join(lines), citations
