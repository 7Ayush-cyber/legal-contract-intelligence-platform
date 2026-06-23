# Legal Contract Intelligence Platform

An enterprise-style Legal AI platform for contract question answering, clause extraction, risk detection, missing clause analysis, and contract comparison.

Built on top of the CUAD legal contract dataset, this project combines hybrid retrieval with contract analysis agents to turn dense legal text into clear, structured, and actionable outputs.

## Live Demo

**Streamlit App:** <https://legal-contract-intelligence-platform-4gnq36xirwvyageshtrqcc.streamlit.app/>

## Key Features

### Contract Q&A
Ask legal questions across a large contract corpus and get grounded answers with supporting sources.

Example questions:
- What confidentiality obligations exist?
- What governing law provisions exist?
- What termination rights exist?
- What force majeure clauses exist?

### Contract Analysis
Upload a single contract and generate:
- Clause Extraction
- Risk Detection
- Missing Clause Detection
- Executive Summary
- Final Legal Report

### Contract Comparison
Upload two contracts and compare them across:
- Purpose
- Confidentiality
- Termination
- Force Majeure
- Indemnification
- Governing Law
- Intellectual Property
- Assignment
- Dispute Resolution
- Liability Allocation

## How It Works

The platform has two major backend systems:

### 1. RAG Backend
Used for legal question answering across the contract corpus.

It combines:
- FAISS for semantic retrieval
- BM25 for keyword retrieval
- Hybrid fusion
- LLM reranking
- Self-RAG style answering

### 2. Legal Analysis Backend
Used for single-contract analysis and comparison.

It performs:
- Clause extraction
- Risk identification
- Missing clause detection
- Summary generation
- Contract comparison
- Legal report generation

## Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI / Bedrock-compatible LLM
- FAISS
- BM25
- Pandas
- PyPDF
- python-docx
- CUAD dataset

## Project Structure

```text
legal-contract-intelligence-platform/
├── app.py
├── rag_backend.py
├── analysis_backend.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── metadata/
│   └── contract_type_mapping.csv
├── vectorstores/
│   ├── bm25.pkl
│   ├── chunks.pkl
│   └── legal_faiss/
│       ├── index.faiss
│       └── index.pkl
└── notebooks/
    ├── 01_cuad_exploration.ipynb
    ├── 02_legal_rag_baseline.ipynb
    ├── 03_legal_rag_evaluation.ipynb
    ├── 04_self_rag_verification.ipynb
    ├── 05_hybrid_retrieval.ipynb
    ├── 06_advanced_retrieval_reranking.ipynb
    ├── 07_hybrid_legal_self_rag.ipynb
    └── 08_legal_contract_analysis_agent.ipynb
