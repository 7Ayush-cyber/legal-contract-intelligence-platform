# rag_backend.py
# Clean backend for Notebook 07 -> Streamlit

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()


import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in .env"
    )

if not OPENAI_BASE_URL:
    raise ValueError(
        "OPENAI_BASE_URL not found in .env"
    )



import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi  # Needed to unpickle BM25 objects



# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
VECTOR_DIR = PROJECT_DIR / "vectorstores"
METADATA_DIR = PROJECT_DIR / "metadata"
FAISS_DIR = VECTOR_DIR / "legal_faiss"

CHUNKS_PATH = VECTOR_DIR / "chunks.pkl"
BM25_PATH = VECTOR_DIR / "bm25.pkl"
METADATA_PATH = METADATA_DIR / "contract_type_mapping.csv"


# =============================================================================
# LLM + VECTORSTORE LOADING
# =============================================================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

faiss_db = FAISS.load_local(
    FAISS_DIR,
    embeddings,
    allow_dangerous_deserialization=True,
)

with open(BM25_PATH, "rb") as f:
    bm25_retriever = pickle.load(f)

metadata_df = pd.read_csv(METADATA_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

llm = ChatOpenAI(
    model="openai.gpt-oss-120b",
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)


# =============================================================================
# SMALL HELPERS
# =============================================================================

def _serialize_sources(docs: List[Any]) -> List[Dict[str, str]]:
    """Convert retrieved docs into a clean source list for the UI."""
    sources = []
    for doc in docs:
        sources.append(
            {
                "source": doc.metadata.get("source", "Unknown"),
                "contract_type": doc.metadata.get("contract_type", "Unknown"),
            }
        )
    return sources


def _bm25_search(query: str, k: int = 20):
    """
    Make BM25 retrieval a bit more robust in case the pickle contains
    different retriever types.
    """
    if hasattr(bm25_retriever, "invoke"):
        results = bm25_retriever.invoke(query)
        return results[:k] if isinstance(results, list) else list(results)[:k]

    if hasattr(bm25_retriever, "get_relevant_documents"):
        return bm25_retriever.get_relevant_documents(query)[:k]

    raise TypeError(
        "Unsupported BM25 retriever object. Expected a retriever with .invoke() "
        "or .get_relevant_documents()."
    )


def _normalize_ranked_indices(response: str, n_docs: int, top_k: int) -> List[int]:
    """
    Parse LLM reranker output robustly.
    Supports either 0-based or 1-based indexing.
    """
    try:
        raw = [int(x.strip()) for x in response.split(",") if x.strip()]
    except Exception:
        return list(range(min(top_k, n_docs)))

    if not raw:
        return list(range(min(top_k, n_docs)))

    # If the model returns 1-based indices, convert to 0-based.
    if all(1 <= i <= n_docs for i in raw) and 0 not in raw:
        raw = [i - 1 for i in raw]

    cleaned = []
    seen = set()

    for idx in raw:
        if 0 <= idx < n_docs and idx not in seen:
            cleaned.append(idx)
            seen.add(idx)

    if not cleaned:
        cleaned = list(range(min(top_k, n_docs)))

    return cleaned[:top_k]


# =============================================================================
# RRF + HYBRID RETRIEVAL
# =============================================================================

def reciprocal_rank_fusion(
    faiss_docs,
    bm25_docs,
    k: int = 60,
):
    """
    Fuse FAISS and BM25 results using Reciprocal Rank Fusion.
    """
    scores = {}

    for rank, doc in enumerate(faiss_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (rank + k)

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (rank + k)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    doc_lookup = {}
    for doc in faiss_docs + bm25_docs:
        doc_lookup[doc.page_content] = doc

    return [doc_lookup[text] for text, _ in ranked]


def hybrid_search(query: str, k: int = 20):
    """
    Retrieve candidates from both FAISS and BM25, then fuse them.
    """
    faiss_results = faiss_db.similarity_search(query, k=k)
    bm25_results = _bm25_search(query, k=k)

    fused_results = reciprocal_rank_fusion(
        faiss_results,
        bm25_results,
    )

    return fused_results[:k]


# =============================================================================
# LLM RERANKING
# =============================================================================

ranking_prompt = ChatPromptTemplate.from_template(
    """
You are a legal retrieval expert.

QUESTION:
{question}

RETRIEVED CHUNKS:

{chunks}

Rank the 5 most relevant chunks for answering the QUESTION.

Consider:

1. Direct relevance
2. Legal clause match
3. Completeness
4. Specificity
5. Whether the chunk helps answer the question

Return ONLY the chunk numbers in ranked order.

Example:

3,7,1,10,2

Do not explain.
Do not provide reasoning.
Do not write words.
Return only comma-separated chunk numbers
"""
)

rerank_chain = ranking_prompt | llm | StrOutputParser()


def batch_llm_rerank(question: str, docs, top_k: int = 5):
    """
    Ask the LLM to rerank the retrieved docs and return the best few.
    """
    formatted_chunks = ""

    for i, doc in enumerate(docs):
        formatted_chunks += f"""

CHUNK {i}

SOURCE:
{doc.metadata.get("source", "Unknown")}

TEXT:
{doc.page_content[:800]}

----------------------------------------
"""

    response = rerank_chain.invoke(
        {
            "question": question,
            "chunks": formatted_chunks,
        }
    )

    ranked_indices = _normalize_ranked_indices(
        response=response,
        n_docs=len(docs),
        top_k=top_k,
    )

    reranked_docs = []
    for idx in ranked_indices:
        reranked_docs.append(docs[idx])

    return reranked_docs


def advanced_retrieval(query: str, retrieve_k: int = 20, final_k: int = 5):
    """
    Full retrieval stack:
    FAISS + BM25 -> RRF -> LLM rerank
    """
    candidates = hybrid_search(query, k=retrieve_k)
    reranked = batch_llm_rerank(query, candidates, top_k=final_k)
    return reranked


# =============================================================================
# QUESTION ROUTING
# =============================================================================

CLAUSE_KEYWORDS = [
    "governing law",
    "termination",
    "termination rights",
    "confidentiality",
    "indemnification",
    "force majeure",
    "intellectual property",
    "liability",
    "notice",
    "breach",
    "assignment",
    "arbitration",
    "payment terms",
    "liquidated damages",
    "warranty",
    "representation",
]

AMBIGUOUS_ENTITY_PATTERNS = [
    "who is the ceo",
    "what is the stock price",
    "who won the fifa world cup",
    "what is the market price",
    "who is the president",
]

def is_clause_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in CLAUSE_KEYWORDS)

def is_entity_ambiguous(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in AMBIGUOUS_ENTITY_PATTERNS)


question_router_prompt = ChatPromptTemplate.from_template(
    """
You are a legal question router.

Classify the question into exactly one of these labels:

- retrieve: the question should be answered using the contract corpus
- direct: the question can be answered without contract evidence
- clarify: the question is underspecified or ambiguous and needs more detail

Rules:
- Clause-style questions about legal provisions should be routed to retrieve, even if the exact agreement is not named.
- Example retrieve questions:
  - What confidentiality obligations exist?
  - What indemnification obligations exist?
  - What governing law clauses exist?
  - What termination rights exist?
  - What force majeure provisions exist?
- Ambiguous entity questions should be routed to clarify.
- Example clarify questions:
  - Who is the CEO?
  - What is the stock price?
  - Which company are you asking about?

Return ONLY one label:
retrieve
direct
clarify

Question:
{question}
"""
)

question_router_chain = question_router_prompt | llm | StrOutputParser()


def route_question(question: str) -> str:
    q = question.lower().strip()

    if is_clause_question(q):
        return "retrieve"

    if is_entity_ambiguous(q):
        return "clarify"

    result = question_router_chain.invoke({"question": question})
    result = result.strip().lower()

    if result not in {"retrieve", "direct", "clarify"}:
        return "clarify"

    return result


# =============================================================================
# DIRECT ANSWER
# =============================================================================

direct_answer_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Answer the question directly and concisely.

Use general knowledge if the question is not about contracts.
If the question is ambiguous, say what detail is missing.
If you are not certain, say you do not know.

Question:
{question}
"""
)

direct_answer_chain = direct_answer_prompt | llm | StrOutputParser()


def generate_direct_answer(question: str) -> str:
    result = direct_answer_chain.invoke({"question": question})
    return result.strip()


# =============================================================================
# QUERY EXPANSION / REWRITE / REVISE
# =============================================================================

intent_expansion_prompt = ChatPromptTemplate.from_template(
    """
You are an expert legal retrieval specialist.

Your job is to convert a user legal question into an optimized retrieval query
for searching legal contracts.

Rules:
1. Preserve the legal intent.
2. Expand legal terminology where useful.
3. Add synonyms commonly found in contracts.
4. Do NOT answer the question.
5. Do NOT ask for clarification.
6. Return ONLY a retrieval query.
7. Keep the query under 20 words.
8. Prefer contract language that would actually appear in legal clauses.

Examples:

Question:
What happens if a party wants out?

Query:
termination rights termination clause cancellation rights early termination agreement termination

Question:
What happens when a contract ends?

Query:
termination clause post termination obligations survival clause expiration agreement termination

Question:
Who owns the intellectual property?

Query:
intellectual property ownership ownership of IP proprietary rights ownership clause

Question:
What confidentiality obligations exist?

Query:
confidentiality clause confidential information non disclosure obligations confidentiality provisions

Question:
What indemnification obligations exist?

Query:
indemnification clause indemnify hold harmless liability obligations indemnity provisions

Question:
What governing law provisions exist?

Query:
governing law clause choice of law jurisdiction venue applicable law dispute jurisdiction

Question:
What force majeure provisions exist?

Query:
force majeure clause acts of god notice requirements excused performance force majeure event

Question:
What assignment rights exist?

Query:
assignment clause transfer rights delegation successor assignment consent transfer agreement

Question:
What dispute resolution provisions exist?

Query:
dispute resolution arbitration mediation governing law litigation venue dispute clause

Question:
{question}

Query:
"""
)

intent_expansion_chain = intent_expansion_prompt | llm | StrOutputParser()


def expand_legal_intent(question: str) -> str:
    result = intent_expansion_chain.invoke({"question": question})
    result = result.strip()

    if not result or "please provide" in result.lower():
        return question

    return result


query_rewrite_prompt = ChatPromptTemplate.from_template(
    """
You are a legal query rewriting system.

Original Question:
{question}

Retrieved Context:
{context}

Your job is to improve retrieval.

Rules:

1. Preserve the user's intent.
2. Never invent clause numbers.
3. Never invent jurisdictions.
4. Never invent parties.
5. Never assume facts not present.
6. Expand legal terminology.
7. Add likely contract language.
8. Make retrieval easier.
9. Return ONE rewritten query only.

Examples:

Question:
What happens if a party wants out?

Rewrite:
termination rights early termination termination clause cancellation rights agreement termination

Question:
Who owns the intellectual property?

Rewrite:
intellectual property ownership ownership clause proprietary rights ownership of inventions

Question:
What governing law provisions exist?

Rewrite:
governing law clause jurisdiction venue choice of law applicable law dispute jurisdiction

Question:
What force majeure provisions exist?

Rewrite:
force majeure clause acts of god notice requirements excused performance force majeure event

Question:
{question}

Rewrite:
"""
)

query_rewrite_chain = query_rewrite_prompt | llm | StrOutputParser()


def rewrite_question(question: str, docs) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    result = query_rewrite_chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )
    return result.strip()


answer_revise_prompt = ChatPromptTemplate.from_template(
    """
You are a legal answer reviser.

Question:
{question}

Retrieved Context:
{context}

Previous Answer:
{answer}

The previous answer was not grounded or not useful.

Revise the answer so that it is fully supported by the context.
If no reliable answer can be made, say:
Information not found in retrieved contracts.

Return only the revised answer.
"""
)

answer_revise_chain = answer_revise_prompt | llm | StrOutputParser()


def revise_answer(question: str, docs, answer: str) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    result = answer_revise_chain.invoke(
        {
            "question": question,
            "context": context,
            "answer": answer,
        }
    )
    return result.strip()


# =============================================================================
# RETRIEVAL / ANSWER / GRADING
# =============================================================================

retrieval_sufficiency_prompt = ChatPromptTemplate.from_template(
    """
You are a legal evidence evaluator.

Question:
{question}

Retrieved Context:
{context}

Decide whether the retrieved context contains sufficient evidence
to answer the question reliably.

Rules:
- If the answer is directly supported, answer YES
- If the evidence is weak, incomplete, or ambiguous, answer NO
- If there are multiple possible answers and the question does not specify which one, answer NO
- Do not guess

Reply ONLY with:
yes
or
no
"""
)

retrieval_sufficiency_chain = retrieval_sufficiency_prompt | llm | StrOutputParser()


def check_retrieval_sufficiency(question: str, docs) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    result = retrieval_sufficiency_chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )
    return result.strip().lower()


answer_prompt = ChatPromptTemplate.from_template(
    """
You are a legal contract assistant.

Answer ONLY using the supplied context.

If the answer is not available,
say:

Information not found in retrieved contracts.

Question:
{question}

Context:
{context}
"""
)

answer_chain = answer_prompt | llm | StrOutputParser()


def generate_answer(question: str, docs) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    return answer_chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )


grounding_prompt = ChatPromptTemplate.from_template(
    """
You are a legal fact checker.

Retrieved Context:
{context}

Generated Answer:
{answer}

Is the answer fully supported
by the retrieved context?

Reply ONLY:

yes

or

no
"""
)

grounding_grader = grounding_prompt | llm | StrOutputParser()


def grade_grounding(answer: str, docs) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    result = grounding_grader.invoke(
        {
            "answer": answer,
            "context": context,
        }
    )
    return result.lower().strip()


answer_grader_prompt = ChatPromptTemplate.from_template(
    """
Question:
{question}

Generated Answer:
{answer}

Does the answer address
the user's question?

Reply ONLY: yes or no
"""
)

answer_grader = answer_grader_prompt | llm | StrOutputParser()


def grade_answer(question: str, answer: str) -> str:
    result = answer_grader.invoke(
        {
            "question": question,
            "answer": answer,
        }
    )
    return result.lower().strip()


# =============================================================================
# MAIN SELF-RAG ORCHESTRATOR
# =============================================================================

def self_rag(question: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Final public entry point for Streamlit.
    Returns only the final response; no internal retry details are printed.
    """
    original_question = question.strip()
    current_question = original_question

    route = route_question(current_question)

    if route == "clarify":
        return {
            "question": original_question,
            "final_question": current_question,
            "answer": "Question is ambiguous or underspecified. Please provide more specific details.",
            "sources": [],
            "status": "clarify_needed",
        }

    if route == "direct":
        answer = generate_direct_answer(current_question)
        return {
            "question": original_question,
            "final_question": current_question,
            "answer": answer,
            "sources": [],
            "status": "direct_answer",
        }

    docs = []
    answer = ""

    for attempt in range(max_retries):
        # Expand the user question into legal retrieval language.
        retrieval_query = expand_legal_intent(current_question)
        if not retrieval_query or not retrieval_query.strip():
            retrieval_query = current_question

        # Hybrid retrieval + reranking
        docs = advanced_retrieval(
            retrieval_query,
            retrieve_k=20,
            final_k=5,
        )

        # Evidence sufficiency check
        sufficiency = check_retrieval_sufficiency(current_question, docs)
        if sufficiency != "yes":
            if attempt < max_retries - 1:
                rewritten_query = rewrite_question(current_question, docs)
                if not rewritten_query or not rewritten_query.strip():
                    rewritten_query = current_question
                current_question = rewritten_query
                continue

            return {
                "question": original_question,
                "final_question": current_question,
                "answer": "Information not found in retrieved contracts.",
                "sources": [],
                "status": "retrieval_failed",
            }

        # Generate answer
        answer = generate_answer(current_question, docs)

        # Grounding check
        grounding = grade_grounding(answer, docs)
        if grounding != "yes":
            if attempt < max_retries - 1:
                revised = revise_answer(current_question, docs, answer)
                if revised:
                    answer = revised

                grounding = grade_grounding(answer, docs)
                if grounding == "yes":
                    relevance = grade_answer(current_question, answer)
                    if relevance == "yes":
                        return {
                            "question": original_question,
                            "final_question": current_question,
                            "answer": answer,
                            "sources": _serialize_sources(docs),
                            "status": "success",
                        }

                # If still not good, rewrite and retry
                current_question = rewrite_question(current_question, docs) or current_question
                continue

            return {
                "question": original_question,
                "final_question": current_question,
                "answer": "Generated answer was not grounded in retrieved evidence.",
                "sources": [],
                "status": "grounding_failed",
            }

        # Relevance check
        relevance = grade_answer(current_question, answer)
        if relevance != "yes":
            if attempt < max_retries - 1:
                current_question = rewrite_question(current_question, docs) or current_question
                continue

            return {
                "question": original_question,
                "final_question": current_question,
                "answer": "Generated answer did not answer the question.",
                "sources": [],
                "status": "answer_failed",
            }

        # Success
        return {
            "question": original_question,
            "final_question": current_question,
            "answer": answer,
            "sources": _serialize_sources(docs),
            "status": "success",
        }

    return {
        "question": original_question,
        "final_question": current_question,
        "answer": "Unable to generate a reliable answer after retries.",
        "sources": [],
        "status": "failed_after_retries",
    }