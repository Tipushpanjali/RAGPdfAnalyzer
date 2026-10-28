# PDF Document Assistant

A fast, lightweight, and completely offline PDF Document Assistant tailored for low-spec personal computers, including machines with only **4 GB of RAM**.

It empowers users to upload multi-topic PDF documents, manage a personal library, and query information with grounded **extractive answering** that cites exact filenames and page numbers.

---

## Key Features

- **Resource Efficient:** Runs comfortably on 4 GB RAM. No neural networks, GPU, or heavy embedding models.
- **100% Offline & Private:** Operates entirely locally. No external APIs or telemetry.
- **Verifiable Citations:** Exact filename and page-level citations for every retrieved answer.
- **Grounded Extractive Answers:** Zero hallucination. Answers are extracted directly from matching sentences in your uploaded PDFs.
- **Persistent Indexing:** Uses SQLite and pre-calculated BM25 indexes stored on disk; documents are indexed once at upload rather than recomputed per query.
- **User Data Isolation:** Keeps documents and chat sessions segregated by username.

---

## System Requirements

- **OS:** Windows 10/11, macOS, or Linux
- **RAM:** Minimum 4 GB RAM
- **Storage:** ~100 MB for Python environment and dependencies
- **Python:** Version 3.9, 3.10, 3.11, or 3.12

---

## Installation & Setup

### Windows (Command Prompt / PowerShell)

1. Open PowerShell or Command Prompt in the project folder:
   ```cmd
   cd path\to\pdf_document_assistant
   ```

2. Create a virtual environment:
   ```cmd
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Command Prompt:**
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **PowerShell:**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```

4. Install the lightweight dependencies:
   ```cmd
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

With your virtual environment activated, run:

```cmd
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## Running the Automated Test Suite

To verify extraction, BM25 indexing, user isolation, citation tracking, and edge-case handling:

```cmd
python -m unittest test_assistant.py
```

---

## Architecture & How It Works

1. **PDF Processing (`pdf_processor.py`):** Uses `pypdf` to extract text and record page numbers. Documents with fewer than 30 readable characters are flagged as scanned files requiring OCR.
2. **Metadata & Storage (`database.py`):** SQLite maintains document records, text chunks, and isolated chat logs.
3. **Retrieval & Extractive Answering (`retrieval.py`):** 
   - A `BM25Okapi` index is generated per user and serialized to the `indexes/` directory.
   - When a query is entered, BM25 scores passages. Sentences with the highest lexical alignment are compiled into an answer, with the source filename and page number appended.
   - If the BM25 relevance score falls below the required threshold, the application returns: `"I could not find an answer in your uploaded documents."`

---

## Limitations

1. **Purely Extractive (No LLM):** Answers are composed directly of exact sentences and passages found in your PDFs. It will not generate creative summaries or infer complex abstractions not stated in the text.
2. **Scanned Documents (OCR):** PDFs containing raw scanned images without a selectable text layer will display an **"OCR Required"** notice. An external OCR tool (such as OCRmyPDF or Tesseract) should be run on the file before uploading.
3. **Keyword Sensitivity:** Queries yield the most relevant results when they share terminology with the uploaded documents.
