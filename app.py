"""Streamlit user interface for PDF Document Assistant.

Provides user dashboard, multi-file upload, document manager,
and verifiable extractive Q&A chat with page citations.
"""

import json
import os
import streamlit as st

import database as db
import pdf_processor as processor
from retrieval import UserBM25Index, extract_answer

UPLOAD_DIR = "uploads"
INDEX_DIR = "indexes"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

db.init_db()

st.set_page_config(
    page_title="PDF Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #fcfcfc;
    }
    .stMetric {
        background-color: #f7f9fa;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-ready { background-color: #dcfce7; color: #166534; }
    .badge-warning { background-color: #fef9c3; color: #854d0e; }
    .badge-error { background-color: #fee2e2; color: #991b1b; }
    </style>
    """,
    unsafe_allow_html=True
)

if "username" not in st.session_state:
    st.session_state.username = "default_user"

if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = UserBM25Index(st.session_state.username, cache_dir=INDEX_DIR)
    if not st.session_state.bm25_index.load_from_cache():
        user_chunks = db.get_all_user_chunks(st.session_state.username)
        st.session_state.bm25_index.build_or_reload(user_chunks)


def reload_active_index():
    user_chunks = db.get_all_user_chunks(st.session_state.username)
    st.session_state.bm25_index.build_or_reload(user_chunks)


with st.sidebar:
    st.title("👤 User Profile")
    current_user_input = st.text_input("Active Username", value=st.session_state.username).strip()
    if current_user_input and current_user_input != st.session_state.username:
        st.session_state.username = current_user_input
        db.ensure_user(st.session_state.username)
        st.session_state.bm25_index = UserBM25Index(st.session_state.username, cache_dir=INDEX_DIR)
        reload_active_index()
        st.rerun()

    db.ensure_user(st.session_state.username)
    st.markdown("---")

    st.subheader("⚡ Architecture & Status")
    st.markdown(
        """
        - **Engine:** Python + rank-bm25 (Offline)
        - **LLM/Cloud API:** None (100% Private)
        - **Target Spec:** 4 GB RAM / Low-spec CPU
        - **Answering:** Extractive (Exact Source)
        """
    )
    st.info(
        "💡 **Notice:** Scanned documents without digital text layers will not return matches unless processed via OCR beforehand."
    )

tab_dashboard, tab_chat, tab_about = st.tabs(["📊 Document Library & Dashboard", "💬 Ask Documents", "ℹ️ System Info & Limitations"])

with tab_dashboard:
    st.header("Document Dashboard")
    st.caption("Upload PDFs, inspect status, and manage your isolated document collection.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Files are processed and indexed locally."
        )

        if uploaded_files:
            if st.button("Process & Index Files", type="primary"):
                user_folder = os.path.join(UPLOAD_DIR, st.session_state.username)
                os.makedirs(user_folder, exist_ok=True)

                total_files = len(uploaded_files)
                progress_bar = st.progress(0)

                for idx, uploaded_file in enumerate(uploaded_files):
                    dest_path = os.path.join(user_folder, uploaded_file.name)
                    with open(dest_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    status_msg, pages, chunks = processor.process_pdf(dest_path, uploaded_file.name)
                    doc_id = db.add_document(
                        username=st.session_state.username,
                        filename=uploaded_file.name,
                        filepath=dest_path,
                        total_pages=pages,
                        chunks_count=len(chunks),
                        status=status_msg
                    )

                    if chunks:
                        for chunk in chunks:
                            chunk["doc_id"] = doc_id
                            chunk["username"] = st.session_state.username
                        db.add_chunks_batch(chunks)

                    progress_bar.progress((idx + 1) / total_files)

                reload_active_index()
                st.success("All uploaded documents processed and indexed successfully!")
                st.rerun()

    with col2:
        st.subheader("Library Metrics")
        user_docs = db.get_user_documents(st.session_state.username)
        total_docs = len(user_docs)
        total_pages_all = sum(d["total_pages"] for d in user_docs)
        total_chunks_all = sum(d["extracted_chunks"] for d in user_docs)

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Documents", total_docs)
        mcol2.metric("Total Pages", total_pages_all)
        mcol3.metric("Indexed Passages", total_chunks_all)

    st.markdown("---")
    st.subheader("Your Documents")

    search_term = st.text_input("Filter documents by filename", placeholder="Type to filter...")
    filtered_docs = db.get_user_documents(st.session_state.username, search_query=search_term)

    if not filtered_docs:
        st.write("No documents found in your library.")
    else:
        for doc in filtered_docs:
            with st.container():
                dcol1, dcol2, dcol3, dcol4 = st.columns([3, 1, 3, 1])
                dcol1.markdown(f"**📄 {doc['filename']}**")
                dcol2.write(f"{doc['total_pages']} page(s)")
                
                if "Ready" in doc["status"]:
                    dcol3.markdown(f"<span class='badge badge-ready'>Ready ({doc['extracted_chunks']} chunks)</span>", unsafe_allow_html=True)
                elif "OCR" in doc["status"] or "scanned" in doc["status"]:
                    dcol3.markdown("<span class='badge badge-warning'>OCR Required</span>", unsafe_allow_html=True)
                else:
                    dcol3.markdown(f"<span class='badge badge-error'>{doc['status']}</span>", unsafe_allow_html=True)

                if dcol4.button("Delete", key=f"del_{doc['id']}"):
                    deleted_path = db.delete_document(doc["id"], st.session_state.username)
                    if deleted_path and os.path.exists(deleted_path):
                        try:
                            os.remove(deleted_path)
                        except OSError:
                            pass
                    reload_active_index()
                    st.toast(f"Deleted {doc['filename']}")
                    st.rerun()
                st.divider()


with tab_chat:
    st.header("Document Question Answering")

    st.caption(
        "Search for document sections, headings, or specific information. "
        "Matching sections are extracted directly from your PDFs."
    )

    chat_history = db.get_user_chat_history(st.session_state.username)

    top_col1, top_col2 = st.columns([6, 1])

    with top_col2:
        if st.button("Clear Chat", help="Clear chat history for current user"):
            db.clear_user_chat_history(st.session_state.username)
            st.rerun()

    # Display previous conversations.
    for message in chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["message"])

            if message["citations_json"]:
                try:
                    citations = json.loads(message["citations_json"])
                except (json.JSONDecodeError, TypeError):
                    citations = []

                if citations:
                    with st.expander("View source documents and extracted passages"):
                        for citation in citations:

                            st.markdown(
                                f"**Source:** `{citation['filename']}`"
                            )

                            st.markdown(
                                f"**Page:** {citation['page_number']}"
                            )

                            st.markdown(
                                f"> {citation.get('passage', '')}"
                            )

                            st.divider()

    # Accept a new question.
    user_query = st.chat_input(
        "Try: Introduction, Objectives, Methodology, or Conclusion..."
    )

    if user_query:

        with st.chat_message("user"):
            st.markdown(user_query)

        db.save_chat_message(
            st.session_state.username,
            "user",
            user_query
        )

        with st.chat_message("assistant"):

            with st.spinner("Searching your documents..."):

                retrieved = st.session_state.bm25_index.query(
                    user_query,
                    top_k=5
                )

                answer, citations = extract_answer(
                    user_query,
                    retrieved
                )

            if citations:

                st.success("Relevant content found in your documents.")

                st.markdown(answer)

                with st.expander("View verified source documents"):

                    for citation in citations:

                        st.markdown(
                            f"**Source:** `{citation['filename']}`"
                        )

                        st.markdown(
                            f"**Page:** {citation['page_number']}"
                        )

                        st.markdown(
                            f"> {citation.get('passage', '')}"
                        )

                        st.divider()

            else:

                st.warning("No sufficiently relevant content was found.")

                st.markdown(answer)

        db.save_chat_message(
            st.session_state.username,
            "assistant",
            answer,
            citations_json=json.dumps(citations)
        )

    st.header("Document Question Answering")
    st.caption("Ask questions about your uploaded documents. Answers are strictly extracted with page citations.")

    chat_history = db.get_user_chat_history(st.session_state.username)

    top_col1, top_col2 = st.columns([6, 1])
    with top_col2:
        if st.button("Clear Chat", help="Clear chat history for current user"):
            db.clear_user_chat_history(st.session_state.username)
            st.rerun()

    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])
            if message["citations_json"]:
                citations = json.loads(message["citations_json"])
                if citations:
                    with st.expander("🔍 View Verified Source Citations & Passages"):
                        for c in citations:
                            st.markdown(f"**Source:** `{c['filename']}` — **Page {c['page_number']}**")
                            st.markdown(f"> {c.get('passage', '')}")
                            st.write("---")

    user_query = st.chat_input("Ask a question based on your uploaded PDFs...")
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        db.save_chat_message(st.session_state.username, "user", user_query)

        with st.spinner("Searching indexed documents..."):
            retrieved = st.session_state.bm25_index.query(user_query, top_k=5)
            answer, citations = extract_answer(user_query, retrieved)

        with st.chat_message("assistant"):
            st.markdown(answer)
            if citations:
                with st.expander("🔍 View Verified Source Citations & Passages"):
                    for c in citations:
                        st.markdown(f"**Source:** `{c['filename']}` — **Page {c['page_number']}**")
                        st.markdown(f"> {c.get('passage', '')}")
                        st.write("---")

        db.save_chat_message(
            st.session_state.username,
            "assistant",
            answer,
            citations_json=json.dumps(citations)
        )

with tab_about:
    st.header("About PDF Document Assistant")
    
st.markdown(
    """
    ### Principles of Operation

    1. **Low Memory Design:** Uses Python and BM25 for offline document retrieval.
    2. **100% Offline & Private:** No external language model or cloud API is required.
    3. **Structure-Aware Extraction:** Preserves headings and paragraph boundaries.
    4. **Section-Based Retrieval:** Matches exact and similar headings and extracts their associated content.
    5. **Source Verification:** Displays original PDF filenames and page references.

    ### Search Capabilities

    - Exact heading searches.
    - Similar heading searches.
    - Complete section extraction.
    - Paragraph-level BM25 fallback.
    - Source and page references.

    ### Important Limitations

    - **Scanned PDFs:** Documents without selectable text require OCR.
    - **Heading Detection:** Unusual PDF layouts may affect heading recognition.
    - **Extractive Search:** The application extracts original document content rather than generating new explanations.
    - **Keyword Search:** When no suitable heading exists, paragraph retrieval depends on matching terminology.
    """
)
