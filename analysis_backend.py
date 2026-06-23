# analysis_backend.py
# Clean backend for Notebook 08 -> Streamlit

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

if not OPENAI_BASE_URL:
    raise ValueError("OPENAI_BASE_URL not found in .env")

# =============================================================================
# PROJECT PATHS
# =============================================================================
# Put this file in: legal_ai/backend/analysis_backend.py
# If you keep it at project root, change parents[1] to parent.

PROJECT_DIR = Path(__file__).resolve().parent
VECTOR_DIR = PROJECT_DIR / "vectorstores"
METADATA_DIR = PROJECT_DIR / "metadata"
CHUNKS_PATH = VECTOR_DIR / "chunks.pkl"
METADATA_PATH = METADATA_DIR / "contract_type_mapping.csv"

# =============================================================================
# DATA LOADING
# =============================================================================

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

metadata_df = pd.read_csv(METADATA_PATH)
metadata_df["source"] = metadata_df["source"].astype(str)

llm = ChatOpenAI(
    model="openai.gpt-oss-120b",
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# =============================================================================
# SHARED GUARDRAILS
# =============================================================================

STRICT_GUARDRAILS = """
IMPORTANT RULES:
- Use only the supplied contract text.
- Do not assume facts not present.
- Do not infer clauses that are not explicitly supported.
- Do not use external knowledge.
- If evidence is absent, say: "Not found in supplied text."
- Keep answers grounded in the provided text.
"""

# =============================================================================
# HELPERS
# =============================================================================

def get_contract_text(source_name: str, max_chunks: int = 20) -> str:
    """
    Reconstruct a contract from its chunks using the source filename.
    """
    contract_chunks = [
        chunk.page_content
        for chunk in chunks
        if chunk.metadata.get("source") == source_name
    ]

    if not contract_chunks:
        raise ValueError(f"No chunks found for source: {source_name}")

    return "\n\n".join(contract_chunks[:max_chunks])


# =============================================================================
# CLAUSE EXTRACTION AGENT
# =============================================================================

clause_extraction_prompt = ChatPromptTemplate.from_template(
    """
You are a senior legal contract reviewer.

{guardrails}

Analyze the provided contract carefully.

For EACH clause below determine whether it is:

- Present
- Referenced
- Partially Present
- Missing

Definitions:

Present:
The clause appears directly and substantially in the provided text.

Referenced:
The clause does not appear directly, but the document clearly incorporates
or references another agreement where the clause likely exists.

Partially Present:
Only part of the clause appears, or the clause is incomplete.

Missing:
The clause is neither present nor referenced.

Review the following clauses:

1. Confidentiality
2. Termination
3. Force Majeure
4. Indemnification
5. Governing Law
6. Intellectual Property
7. Assignment
8. Dispute Resolution

For each clause provide:

| Clause | Status | Evidence | Key Obligations | Legal Risk | Explanation |

Risk Levels:
Low / Medium / High

If evidence is unavailable write:
"Not found in supplied text."

Contract:
{contract}

Return ONLY a professional markdown table.
"""
)

clause_extraction_chain = clause_extraction_prompt | llm | StrOutputParser()

def extract_clauses(contract_text: str) -> str:
    return clause_extraction_chain.invoke(
        {
            "guardrails": STRICT_GUARDRAILS,
            "contract": contract_text,
        }
    )

def clause_agent(contract_text: str) -> str:
    return extract_clauses(contract_text)


# =============================================================================
# RISK DETECTION AGENT
# =============================================================================

risk_detection_prompt = ChatPromptTemplate.from_template(
    """
You are a senior legal risk analyst.

{guardrails}

Review the contract text and identify ONLY risks supported by the text.

Focus on:
- Missing confidentiality clause
- Missing termination rights
- Missing force majeure protection
- Missing indemnification
- Missing governing law
- Missing assignment restrictions
- Missing dispute resolution mechanism
- One-sided obligations
- Unlimited liability
- Weak protection of confidential information
- Weak IP ownership language

For each risk, provide:
1. Risk category
2. Why it is risky
3. Severity: Low / Medium / High
4. Practical recommendation

Contract text:
{contract}

Return structured markdown.
"""
)

risk_detection_chain = risk_detection_prompt | llm | StrOutputParser()

def detect_risks(contract_text: str) -> str:
    return risk_detection_chain.invoke(
        {
            "guardrails": STRICT_GUARDRAILS,
            "contract": contract_text,
        }
    )

def risk_agent(contract_text: str) -> str:
    return detect_risks(contract_text)


# =============================================================================
# MISSING CLAUSE DETECTION AGENT
# =============================================================================

missing_clause_prompt = ChatPromptTemplate.from_template(
    """
You are a senior legal contract reviewer.

{guardrails}

Review the contract.

For each clause below determine:

- Present
- Referenced
- Partially Present
- Missing

Clauses:

1. Confidentiality
2. Termination
3. Force Majeure
4. Indemnification
5. Governing Law
6. Intellectual Property
7. Assignment
8. Dispute Resolution

Definitions:

Present:
Clause directly appears.

Referenced:
Clause likely exists in an incorporated agreement or referenced document.

Partially Present:
Incomplete version appears.

Missing:
No evidence of clause.

For every clause provide:

| Clause | Status | Why It Matters | Potential Risk | Recommendation |

Contract:
{contract}

Return structured markdown.
"""
)

missing_clause_chain = missing_clause_prompt | llm | StrOutputParser()

def detect_missing_clauses(contract_text: str) -> str:
    return missing_clause_chain.invoke(
        {
            "guardrails": STRICT_GUARDRAILS,
            "contract": contract_text,
        }
    )

def missing_clause_agent(contract_text: str) -> str:
    return detect_missing_clauses(contract_text)


# =============================================================================
# SUMMARY AGENT
# =============================================================================

summary_prompt = ChatPromptTemplate.from_template(
    """
You are a legal executive advisor.

{guardrails}

Write a concise but useful summary of the contract.

Include:
1. Purpose of the contract
2. Key obligations
3. Important risks
4. Important missing clauses
5. Practical recommendations

Write in clear business language.

Contract text:
{contract}

Return structured markdown.
"""
)

summary_chain = summary_prompt | llm | StrOutputParser()

def generate_summary(contract_text: str) -> str:
    return summary_chain.invoke(
        {
            "guardrails": STRICT_GUARDRAILS,
            "contract": contract_text,
        }
    )

def summary_agent(contract_text: str) -> str:
    return generate_summary(contract_text)


# =============================================================================
# CONTRACT COMPARISON AGENT
# =============================================================================

comparison_prompt = ChatPromptTemplate.from_template(
"""
You are a senior legal contracts expert.

{guardrails}

IMPORTANT INSTRUCTIONS

- Use ONLY the text provided in Contract A and Contract B.
- Do NOT use outside legal knowledge.
- Do NOT assume a clause exists if it is not found in the provided text.
- If a clause is not found, write:
  "Not found in provided text."
- Do NOT state that a clause is missing from the full contract unless the text clearly proves that.
- Base all conclusions only on the supplied excerpts.
- If evidence is insufficient, explicitly say:
  "Insufficient evidence in provided text."

------------------------------------------------------------------

Compare Contract A and Contract B.

Focus on the following categories:

1. Contract Purpose
2. Confidentiality
3. Termination
4. Force Majeure
5. Indemnification
6. Governing Law
7. Intellectual Property
8. Assignment
9. Dispute Resolution
10. Liability Allocation

For EACH category provide:

### Category Name

Present in Contract A:
- Yes / No / Partially Present / Not Found in Provided Text

Present in Contract B:
- Yes / No / Partially Present / Not Found in Provided Text

Evidence:
- Briefly cite the relevant language or clause description.

Major Differences:
- Explain the practical differences.

Risk Assessment:
- Explain the legal or commercial risks created by those differences.

Protection Assessment:
- Based ONLY on the provided text,
  which contract appears to provide stronger protection?
- If uncertain, state:
  "Insufficient evidence in provided text."

------------------------------------------------------------------

After all categories provide:

# EXECUTIVE COMPARISON

Provide a business-friendly overview of the most important findings.

------------------------------------------------------------------

# KEY DIFFERENCES

Summarize the top differences between the contracts.

------------------------------------------------------------------

# MAJOR RISKS

List the most significant legal and commercial risks identified.

For each risk include:

- Risk
- Severity (Low / Medium / High)
- Reason

------------------------------------------------------------------

# RECOMMENDATIONS

Provide practical recommendations for improving each contract.

Examples:
- Add governing law clause
- Add force majeure protection
- Add indemnification language
- Add assignment restrictions
- Add dispute resolution mechanism
- Clarify IP ownership
- Add limitation of liability

------------------------------------------------------------------

# FINAL CONCLUSION

Provide a concise conclusion:

- Which contract appears stronger based on the provided text?
- Why?
- What are the most important improvements required?

Contract A:
{contract_a}

Contract B:
{contract_b}

Return professional markdown suitable for a legal review report.
"""
)

comparison_chain = comparison_prompt | llm | StrOutputParser()

def compare_contracts(contract_a: str, contract_b: str) -> str:
    return comparison_chain.invoke(
        {
            "guardrails": STRICT_GUARDRAILS,
            "contract_a": contract_a,
            "contract_b": contract_b,
        }
    )

def comparison_agent(contract_a: str, contract_b: str) -> str:
    return compare_contracts(contract_a, contract_b)


# =============================================================================
# LEGAL REPORT GENERATION
# =============================================================================

legal_report_prompt = ChatPromptTemplate.from_template(
    """
You are a senior legal advisor.

{guardrails}

Create a professional legal report.

Clause Analysis:
{clauses}

Risk Analysis:
{risks}

Missing Clauses:
{missing}

Summary:
{summary}

Produce:

1. Executive Summary
2. Major Clauses
3. Key Risks
4. Missing Clauses
5. Recommendations

Use professional legal language.
"""
)

legal_report_chain = legal_report_prompt | llm | StrOutputParser()

def generate_legal_report(
    clauses: str,
    risks: str,
    missing: str,
    summary: str,
) -> str:
    return legal_report_chain.invoke(
        {
            "guardrails": STRICT_GUARDRAILS,
            "clauses": clauses,
            "risks": risks,
            "missing": missing,
            "summary": summary,
        }
    )

def report_agent(
    clauses: str,
    risks: str,
    missing: str,
    summary: str,
) -> str:
    return generate_legal_report(clauses, risks, missing, summary)


# =============================================================================
# FULL CONTRACT ANALYSIS
# =============================================================================

def analyze_contract(contract_text: str) -> Dict[str, str]:
    clauses = extract_clauses(contract_text)
    risks = detect_risks(contract_text)
    missing = detect_missing_clauses(contract_text)
    summary = generate_summary(contract_text)
    report = generate_legal_report(clauses, risks, missing, summary)

    return {
        "clauses": clauses,
        "risks": risks,
        "missing": missing,
        "missing_clauses": missing,  # backward compatibility
        "summary": summary,
        "report": report,
    }


# =============================================================================
# PUBLIC AGENT ENTRY POINTS
# =============================================================================

def legal_contract_agent(task_or_contract_text, contract_text=None, contract_b=None):
    """
    Backward-compatible dispatcher for Streamlit and older notebook calls.

    Supported tasks:
    - analysis / full_analysis / analyze
    - clauses
    - risks
    - missing / missing_clauses
    - summary
    - compare / comparison

    Old style:
    - legal_contract_agent(contract_text)
    """
    known_tasks = {
        "analysis",
        "full_analysis",
        "analyze",
        "clauses",
        "risks",
        "missing",
        "missing_clauses",
        "summary",
        "compare",
        "comparison",
    }

    # Old-style call: legal_contract_agent(contract_text)
    if (
        contract_text is None
        and contract_b is None
        and isinstance(task_or_contract_text, str)
        and task_or_contract_text.strip().lower() not in known_tasks
    ):
        return analyze_contract(task_or_contract_text)

    task = str(task_or_contract_text or "").strip().lower()

    if task in {"analysis", "full_analysis", "analyze"}:
        if contract_text is None:
            raise ValueError("contract_text is required for full analysis.")
        return analyze_contract(contract_text)

    if task == "clauses":
        if contract_text is None:
            raise ValueError("contract_text is required for clause extraction.")
        return extract_clauses(contract_text)

    if task == "risks":
        if contract_text is None:
            raise ValueError("contract_text is required for risk detection.")
        return detect_risks(contract_text)

    if task in {"missing", "missing_clauses"}:
        if contract_text is None:
            raise ValueError("contract_text is required for missing clause detection.")
        return detect_missing_clauses(contract_text)

    if task == "summary":
        if contract_text is None:
            raise ValueError("contract_text is required for summary generation.")
        return generate_summary(contract_text)

    if task in {"compare", "comparison"}:
        if contract_text is None or contract_b is None:
            raise ValueError("Both contract_text and contract_b are required for comparison.")
        return compare_contracts(contract_text, contract_b)

    raise ValueError(
        "Unsupported task. Use one of: analysis, clauses, risks, missing, summary, compare."
    )

def legal_comparison_agent(contract_a: str, contract_b: str) -> Dict[str, str]:
    comparison_report = compare_contracts(contract_a, contract_b)
    return {
        "comparison_report": comparison_report,
        "report": comparison_report,
    }

__all__ = [
    "chunks",
    "metadata_df",
    "get_contract_text",
    "extract_clauses",
    "detect_risks",
    "detect_missing_clauses",
    "generate_summary",
    "compare_contracts",
    "generate_legal_report",
    "analyze_contract",
    "legal_contract_agent",
    "legal_comparison_agent",
]







if __name__ == "__main__":

    source = metadata_df.iloc[0]["source"]

    contract = get_contract_text(
        source,
        max_chunks=20
    )

    result = analyze_contract(
        contract
    )

    print("=" * 100)
    print("LEGAL REPORT")
    print("=" * 100)

    print(result["report"][:3000])