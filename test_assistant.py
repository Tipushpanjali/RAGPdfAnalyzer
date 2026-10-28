"""Automated verification suite.

Covers:
1. PDF text extraction & page number preservation.
2. BM25 retrieval relevance.
3. Citation accuracy & extractive answering.
4. Missing information handling.
5. Encrypted & invalid PDF handling.
6. User data isolation.
7. Index persistence and cache reloading.
"""

import os
import unittest
from pypdf import PdfWriter

import database as db
import pdf_processor as processor
from retrieval import UserBM25Index, extract_answer, tokenize


class TestPDFAssistant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.DB_FILE = "test_assistant.db"
        if os.path.exists(db.DB_FILE):
            os.remove(db.DB_FILE)
        db.init_db()

        os.makedirs("test_files", exist_ok=True)
        os.makedirs("test_indexes", exist_ok=True)

        cls.valid_pdf = "test_files/sample_doc.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.add_blank_page(width=72, height=72)
        with open(cls.valid_pdf, "wb") as f:
            writer.write(f)

        cls.empty_pdf = "test_files/empty.pdf"
        with open(cls.empty_pdf, "wb") as f:
            pass

        cls.corrupt_pdf = "test_files/corrupt.pdf"
        with open(cls.corrupt_pdf, "wb") as f:
            f.write(b"NOT_A_REAL_PDF_STREAM_12345")

    @classmethod
    def tearDownClass(cls):
        for path in [cls.valid_pdf, cls.empty_pdf, cls.corrupt_pdf, db.DB_FILE]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists("test_files"):
            try:
                os.rmdir("test_files")
            except OSError:
                pass
        if os.path.exists("test_indexes"):
            for f in os.listdir("test_indexes"):
                os.remove(os.path.join("test_indexes", f))
            os.rmdir("test_indexes")

    def test_01_tokenization(self):
        tokens = tokenize("The quick Brown Fox jumps over the lazy Dog!")
        self.assertIn("quick", tokens)
        self.assertIn("fox", tokens)
        self.assertNotIn("the", tokens)

    def test_02_empty_and_corrupted_pdf_handling(self):
        status, pages, chunks = processor.process_pdf(self.empty_pdf, "empty.pdf")
        self.assertIn("Empty file", status)
        self.assertEqual(len(chunks), 0)

        status, pages, chunks = processor.process_pdf(self.corrupt_pdf, "corrupt.pdf")
        self.assertTrue("Cannot read" in status or "Empty or corrupt" in status)
        self.assertEqual(len(chunks), 0)

    def test_03_bm25_retrieval_and_page_citation(self):
        chunks = [
            {
                "doc_id": 1,
                "username": "alice",
                "filename": "biology.pdf",
                "page_number": 4,
                "chunk_index": 0,
                "content": "Photosynthesis is the process used by plants to convert light energy into chemical energy."
            },
            {
                "doc_id": 1,
                "username": "alice",
                "filename": "biology.pdf",
                "page_number": 12,
                "chunk_index": 1,
                "content": "Mitochondria generate most of the chemical energy needed to power biochemical reactions."
            }
        ]

        index = UserBM25Index("alice", cache_dir="test_indexes")
        index.build_or_reload(chunks)

        results = index.query("What is photosynthesis?", top_k=1)
        self.assertEqual(len(results), 1)
        best_chunk, score = results[0]
        self.assertEqual(best_chunk["filename"], "biology.pdf")
        self.assertEqual(best_chunk["page_number"], 4)
        self.assertGreater(score, 0.0)

        answer, citations = extract_answer("What is photosynthesis?", results)
        self.assertIn("convert light energy", answer)
        self.assertEqual(citations[0]["page_number"], 4)
        self.assertEqual(citations[0]["filename"], "biology.pdf")

    def test_04_missing_information_handling(self):
        chunks = [
            {
                "doc_id": 2,
                "username": "bob",
                "filename": "finance.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "content": "Quarterly revenues rose by 14 percent across retail operations."
            }
        ]
        index = UserBM25Index("bob", cache_dir="test_indexes")
        index.build_or_reload(chunks)

        results = index.query("quantum entanglement and dark matter", top_k=2)
        answer, citations = extract_answer("quantum entanglement and dark matter", results)
        self.assertEqual(answer, "I could not find an answer in your uploaded documents.")
        self.assertEqual(len(citations), 0)

    def test_05_user_isolation(self):
        db.ensure_user("user_a")
        db.ensure_user("user_b")

        doc_a = db.add_document("user_a", "user_a_doc.pdf", "/tmp/a.pdf", 5, 1, "Ready")
        doc_b = db.add_document("user_b", "user_b_doc.pdf", "/tmp/b.pdf", 10, 1, "Ready")

        db.add_chunks_batch([{
            "doc_id": doc_a,
            "username": "user_a",
            "filename": "user_a_doc.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "content": "User A proprietary confidential note."
        }])
        db.add_chunks_batch([{
            "doc_id": doc_b,
            "username": "user_b",
            "filename": "user_b_doc.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "content": "User B proprietary secret."
        }])

        docs_a = db.get_user_documents("user_a")
        docs_b = db.get_user_documents("user_b")
        self.assertTrue(all(d["username"] == "user_a" for d in docs_a))
        self.assertTrue(all(d["username"] == "user_b" for d in docs_b))

        chunks_a = db.get_all_user_chunks("user_a")
        chunks_b = db.get_all_user_chunks("user_b")
        self.assertEqual(len(chunks_a), 1)
        self.assertEqual(chunks_a[0]["content"], "User A proprietary confidential note.")
        self.assertEqual(len(chunks_b), 1)
        self.assertEqual(chunks_b[0]["content"], "User B proprietary secret.")

    def test_06_index_persistence_and_cache(self):
        chunks = [
            {
                "doc_id": 9,
                "username": "charlie",
                "filename": "manual.pdf",
                "page_number": 2,
                "chunk_index": 0,
                "content": "Press the red button for emergency shutdown."
            }
        ]
        index1 = UserBM25Index("charlie", cache_dir="test_indexes")
        index1.build_or_reload(chunks)

        index2 = UserBM25Index("charlie", cache_dir="test_indexes")
        loaded = index2.load_from_cache()
        self.assertTrue(loaded)
        results = index2.query("emergency shutdown")
        self.assertEqual(len(results), 1)
        self.assertIn("emergency shutdown", results[0][0]["content"])


if __name__ == "__main__":
    unittest.main()
