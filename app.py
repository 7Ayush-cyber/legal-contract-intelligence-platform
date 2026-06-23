import streamlit as st
from pathlib import Path
import tempfile
import os

from rag_backend import self_rag
from analysis_backend import (
    legal_contract_agent,
    legal_comparison_agent,
)

try:
    from docx import Document
except ImportError:
    Document = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


st.set_page_config(
    page_title="Legal Contract Intelligence Platform",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Legal Contract Intelligence Platform")
st.caption("Contract Q&A, contract analysis, and contract comparison in one place.")


# ==========================================================
# FILE TEXT EXTRACTION
# ==========================================================

def extract_text_from_txt(uploaded_file) -> str:
    return uploaded_file.read().decode("utf-8", errors="ignore")


def extract_text_from_pdf(uploaded_file) -> str:
    if PdfReader is None:
        raise ImportError("pypdf is not installed.")
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


def extract_text_from_docx(uploaded_file) -> str:
    if Document is None:
        raise ImportError("python-docx is not installed.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        doc = Document(temp_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs).strip()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def extract_uploaded_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)

    raise ValueError("Unsupported file type. Please upload TXT, PDF, or DOCX.")


def show_sources(sources):
    if not sources:
        st.write("No sources returned.")
        return

    for i, src in enumerate(sources, start=1):
        if isinstance(src, dict):
            source_name = src.get("source", "Unknown")
            contract_type = src.get("contract_type", "Unknown")
            st.write(f"{i}. **{source_name}**  \n   *{contract_type}*")
        else:
            st.write(f"{i}. {src}")


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a task",
    ["Contract Q&A", "Contract Analysis", "Contract Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Q&A uses the RAG backend with FAISS + BM25.\n\n"
    "Analysis and comparison use the legal contract analysis backend."
)


# ==========================================================
# PAGE 1: CONTRACT Q&A
# ==========================================================

if page == "Contract Q&A":
    st.subheader("Ask a legal question about the contract corpus")

    question = st.text_area(
        "Type your question",
        placeholder="Example: What governing law provisions exist?",
        height=120,
    )

    ask_btn = st.button("Get Answer", type="primary")

    if ask_btn:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching contracts and generating answer..."):
                result = self_rag(question)

            st.markdown("### Final Answer")
            st.write(result.get("answer", ""))

            st.markdown("### Status")
            st.write(result.get("status", "unknown"))

            st.markdown("### Sources")
            show_sources(result.get("sources", []))


# ==========================================================
# PAGE 2: CONTRACT ANALYSIS
# ==========================================================

elif page == "Contract Analysis":
    st.subheader("Upload one contract for analysis")

    uploaded_file = st.file_uploader(
        "Upload a contract file",
        type=["txt", "pdf", "docx"],
        key="analysis_upload",
    )

    analysis_mode = st.selectbox(
        "Analysis type",
        [
            "Full Analysis",
            "Clause Extraction",
            "Risk Detection",
            "Missing Clauses",
            "Summary",
        ],
    )

    analyze_btn = st.button("Run Analysis", type="primary")

    if analyze_btn:
        if uploaded_file is None:
            st.warning("Please upload a TXT, PDF, or DOCX file.")
        else:
            try:
                with st.spinner("Extracting text from file..."):
                    contract_text = extract_uploaded_text(uploaded_file)

                if not contract_text.strip():
                    st.error("No text could be extracted from the uploaded file.")
                else:
                    with st.spinner("Running legal analysis..."):
                        if analysis_mode == "Full Analysis":
                            result = legal_contract_agent(
                                "full_analysis",
                                contract_text=contract_text,
                            )

                            st.markdown("### Clause Extraction")
                            st.markdown(result["clauses"])

                            st.markdown("### Risk Detection")
                            st.markdown(result["risks"])

                            st.markdown("### Missing Clauses")
                            st.markdown(result["missing"])

                            st.markdown("### Summary")
                            st.markdown(result["summary"])

                            st.markdown("### Final Report")
                            st.markdown(result["report"])

                        elif analysis_mode == "Clause Extraction":
                            result = legal_contract_agent(
                                "clauses",
                                contract_text=contract_text,
                            )
                            st.markdown("### Clause Extraction")
                            st.markdown(result)

                        elif analysis_mode == "Risk Detection":
                            result = legal_contract_agent(
                                "risks",
                                contract_text=contract_text,
                            )
                            st.markdown("### Risk Detection")
                            st.markdown(result)

                        elif analysis_mode == "Missing Clauses":
                            result = legal_contract_agent(
                                "missing",
                                contract_text=contract_text,
                            )
                            st.markdown("### Missing Clauses")
                            st.markdown(result)

                        elif analysis_mode == "Summary":
                            result = legal_contract_agent(
                                "summary",
                                contract_text=contract_text,
                            )
                            st.markdown("### Summary")
                            st.markdown(result)

            except Exception as e:
                st.error(f"Analysis failed: {e}")


# ==========================================================
# PAGE 3: CONTRACT COMPARISON
# ==========================================================

elif page == "Contract Comparison":
    st.subheader("Upload two contracts to compare")

    col1, col2 = st.columns(2)

    with col1:
        file_a = st.file_uploader(
            "Upload Contract A",
            type=["txt", "pdf", "docx"],
            key="contract_a",
        )

    with col2:
        file_b = st.file_uploader(
            "Upload Contract B",
            type=["txt", "pdf", "docx"],
            key="contract_b",
        )

    compare_btn = st.button("Compare Contracts", type="primary")

    if compare_btn:
        if file_a is None or file_b is None:
            st.warning("Please upload both contracts.")
        else:
            try:
                with st.spinner("Extracting text from uploaded files..."):
                    text_a = extract_uploaded_text(file_a)
                    text_b = extract_uploaded_text(file_b)

                if not text_a.strip() or not text_b.strip():
                    st.error("Could not extract text from one or both files.")
                else:
                    with st.spinner("Comparing contracts..."):
                        result = legal_comparison_agent(text_a, text_b)

                    st.markdown("### Comparison Report")
                    st.markdown(result["comparison_report"])

            except Exception as e:
                st.error(f"Comparison failed: {e}")