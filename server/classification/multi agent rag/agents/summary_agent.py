from typing import List, Any
from rag_backend.llm_local import run_llm


def _docs_to_text(docs: List[Any], max_chars: int = 2500) -> str:
    """Convert retriever documents into a truncated text block."""
    chunks = []
    total = 0

    for d in docs:
        text = getattr(d, "page_content", str(d))
        text = text.strip()
        if not text:
            continue

        if total + len(text) > max_chars:
            text = text[: max_chars - total]

        chunks.append(text)
        total += len(text)

        if total >= max_chars:
            break

    return "\n\n---\n\n".join(chunks)


def run_summary_agent(
    question: str,
    gene_answer: str,
    drug_answer: str,
    docs: List[Any]
) -> str:
    """
    Produce a multi-paragraph literature review that synthesizes the gene agent,
    drug agent, and retrieved document evidence.
    """

    context_text = _docs_to_text(docs, max_chars=2500)

    prompt = f"""
You are an oncology researcher writing a high-quality biomedical literature review.

USER QUESTION:
{question}

GENE AGENT FINDINGS:
{gene_answer if gene_answer else "(no gene-specific findings)"}

DRUG AGENT FINDINGS:
{drug_answer if drug_answer else "(no drug-specific findings)"}

RELEVANT EXCERPTS FROM LITERATURE:
{context_text}

TASK:
Write a DETAILED, technical literature review that synthesizes:
- biomarker relevance and gene roles
- molecular and signaling pathways
- drug targets, mechanisms, and resistance
- NSCLC vs colorectal cancer differences (if present)
- mechanistic and preclinical/clinical evidence
- therapeutic implications and future directions

REQUIREMENTS:
- Length: at least 4 paragraphs, ideally 5–7 sentences per paragraph.
- Style: formal scientific tone for a graduate bioinformatics audience.
- Do NOT explain that you are an AI or an agent.
- Do NOT say things like "this is a good synthesis".
- Cite context as (Document 1), (Document 2), etc. when appropriate.

Now write the full literature review.
"""

    # Give the summary agent *more* room than default
    return run_llm(prompt)
