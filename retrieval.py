"""Retrieval and Extractive Answering Engine.

Uses BM25 for passage ranking and an exact extractive scoring algorithm
to pinpoint matching sentences with verified filename and page citations.
Strictly offline with zero external dependencies.
"""

import math
import os
import pickle
import re
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


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> List[str]:
    raw_sentences = re.split(r"(?<=[.?!;])\s+", text)
    clean_sents = [s.strip() for s in raw_sentences if len(s.strip()) > 10]
    return clean_sents if clean_sents else [text.strip()]


class UserBM25Index:
    def __init__(self, username: str, cache_dir: str = "indexes"):
        self.username = username
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f"{username}_bm25.pkl")
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict[str, Any]] = []

    def build_or_reload(self, all_user_chunks: List[Dict[str, Any]]) -> None:
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

        tokenized_corpus = [tokenize(c["content"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f, protocol=pickle.HIGHEST_PROTOCOL)
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

    def query(self, question: str, top_k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        if not self.bm25 or not self.chunks:
            return []

        tokens = tokenize(question)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored_pairs = [(self.chunks[i], float(score)) for i, score in enumerate(scores)]
        scored_pairs = [p for p in scored_pairs if p[1] > 0.0]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]


def extract_answer(
    query_text: str,
    retrieved_passages: List[Tuple[Dict[str, Any], float]],
    score_threshold: float = 0.5
) -> Tuple[str, List[Dict[str, Any]]]:
    if not retrieved_passages:
        return "I could not find an answer in your uploaded documents.", []

    top_score = retrieved_passages[0][1]
    if top_score < score_threshold:
        return "I could not find an answer in your uploaded documents.", []

    q_tokens = set(tokenize(query_text))
    extracted_entries: List[Dict[str, Any]] = []
    seen_sentences = set()

    for chunk, chunk_score in retrieved_passages:
        sentences = split_sentences(chunk["content"])
        sentence_scores = []

        for s in sentences:
            s_tokens = set(tokenize(s))
            if not s_tokens:
                continue
            common = q_tokens.intersection(s_tokens)
            overlap_score = len(common) / (len(q_tokens) + 1e-6)
            if overlap_score > 0:
                sentence_scores.append((s, overlap_score))

        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        for s, s_score in sentence_scores[:2]:
            norm_s = s.lower().strip()
            if norm_s not in seen_sentences and s_score >= 0.15:
                seen_sentences.add(norm_s)
                extracted_entries.append({
                    "sentence": s,
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "score": s_score
                })

    if not extracted_entries:
        best_chunk = retrieved_passages[0][0]
        fallback_text = (
            f"According to **{best_chunk['filename']}** (Page {best_chunk['page_number']}):\n\n"
            f"> \"{best_chunk['content']}\""
        )
        citation = [{
            "filename": best_chunk["filename"],
            "page_number": best_chunk["page_number"],
            "passage": best_chunk["content"]
        }]
        return fallback_text, citation

    lines = ["Based on your documents, here are the relevant findings:\n"]
    citations = []

    for item in extracted_entries[:4]:
        lines.append(f"- \"{item['sentence']}\" — *[{item['filename']}, Page {item['page_number']}]*")
        citations.append({
            "filename": item["filename"],
            "page_number": item["page_number"],
            "passage": item["sentence"]
        })

    formatted_answer = "\n".join(lines)
    return formatted_answer, citations
