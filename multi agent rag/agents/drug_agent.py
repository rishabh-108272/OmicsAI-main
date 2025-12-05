from backend.llm_local import run_llm

DRUG_PROMPT = """
You are a biomedical drug-target evaluation agent.

Your task:
- Identify drug interactions, inhibitors, mechanisms of action, and therapeutic context.
- Emphasize targeted agents (e.g., FASN inhibitors like TVB-2640, EGFR inhibitors, etc.) if present.
- Only use the provided documents. Do not hallucinate.

----------------------
CONTEXT:
{context}
----------------------

Question:
{question}

Write a drug-focused evidence summary of 2-3 short paragraphs.
"""


def _build_context(docs, max_chars: int = 800) -> str:
    """Build a compact context string from retrieved docs."""
    chunks = []
    total = 0
    for d in docs:
        text = (getattr(d, "page_content", "") or "").replace("\n", " ").strip()
        if not text:
            continue

        need = max_chars - total
        if need <= 0:
            break

        chunk = text[:need]
        chunks.append(chunk)
        total += len(chunk)

    return "\n\n".join(chunks)


def run_drug_agent(question: str, docs):
    context = _build_context(docs, max_chars=800)
    prompt = DRUG_PROMPT.format(context=context, question=question)
    return run_llm(prompt)
