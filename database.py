"""Database and local storage manager using SQLite.

Manages users, document records, chunk metadata, and chat history.
Designed for low memory footprint and atomic operations.
"""

import os
import sqlite3
from typing import Any, Dict, List, Optional

DB_FILE = "assistant_data.db"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = get_db_connection()
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                total_pages INTEGER NOT NULL,
                extracted_chunks INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users (username) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                filename TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE,
                FOREIGN KEY (username) REFERENCES users (username) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                citations_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users (username) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_doc_chunks_user ON document_chunks(username);
            CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(username);
            CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_messages(username);
            """
        )
    conn.close()


def ensure_user(username: str) -> None:
    conn = get_db_connection()
    with conn:
        conn.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username.strip(),))
    conn.close()


def add_document(username: str, filename: str, filepath: str, total_pages: int, chunks_count: int, status: str) -> int:
    conn = get_db_connection()
    with conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO documents (username, filename, filepath, total_pages, extracted_chunks, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (username, filename, filepath, total_pages, chunks_count, status),
        )
        doc_id = cursor.lastrowid
    conn.close()
    return doc_id


def add_chunks_batch(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        return
    conn = get_db_connection()
    with conn:
        conn.executemany(
            """
            INSERT INTO document_chunks (doc_id, username, filename, page_number, chunk_index, content)
            VALUES (:doc_id, :username, :filename, :page_number, :chunk_index, :content)
            """,
            chunks,
        )
    conn.close()


def get_user_documents(username: str, search_query: str = "") -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    if search_query.strip():
        q = f"%{search_query.strip()}%"
        cursor.execute(
            """
            SELECT * FROM documents
            WHERE username = ? AND filename LIKE ?
            ORDER BY created_at DESC
            """,
            (username, q),
        )
    else:
        cursor.execute(
            """
            SELECT * FROM documents
            WHERE username = ?
            ORDER BY created_at DESC
            """,
            (username,),
        )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def delete_document(doc_id: int, username: str) -> Optional[str]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filepath FROM documents WHERE id = ? AND username = ?", (doc_id, username))
    row = cursor.fetchone()
    filepath = row["filepath"] if row else None

    if filepath:
        with conn:
            cursor.execute("DELETE FROM document_chunks WHERE doc_id = ? AND username = ?", (doc_id, username))
            cursor.execute("DELETE FROM documents WHERE id = ? AND username = ?", (doc_id, username))

    conn.close()
    return filepath


def get_all_user_chunks(username: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, doc_id, filename, page_number, chunk_index, content
        FROM document_chunks
        WHERE username = ?
        ORDER BY id ASC
        """,
        (username,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def save_chat_message(username: str, role: str, message: str, citations_json: str = "") -> None:
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO chat_messages (username, role, message, citations_json)
            VALUES (?, ?, ?, ?)
            """,
            (username, role, message, citations_json),
        )
    conn.close()


def get_user_chat_history(username: str, limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, message, citations_json, created_at
        FROM chat_messages
        WHERE username = ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (username, limit),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def clear_user_chat_history(username: str) -> None:
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM chat_messages WHERE username = ?", (username,))
    conn.close()
